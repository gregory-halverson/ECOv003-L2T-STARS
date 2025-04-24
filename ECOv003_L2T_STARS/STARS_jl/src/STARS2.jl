# module STARS

# export coarse_fine_scene_fusion
# # export scene_fusion
# # 
# export STARSInstrumentData
# export STARSInstrumentGeoData

# Write your package code here.
using Dates
using Rasters
using LinearAlgebra
using Distributions
using Statistics
using StatsBase
using SparseArrays
using BlockDiagonals
using Distances
import GaussianRandomFields.CovarianceFunction
import GaussianRandomFields.Matern
import GaussianRandomFields.apply
using GeoArrays
using MultivariateStats
using Kronecker
using ProgressMeter
using Random
using Interpolations
using KernelFunctions
#BLAS.set_num_threads(1)

include("resampling_utils.jl")
include("spatial_utils_ll.jl")
include("GP_utils.jl")

T = Float64

struct KSModel{Float64}
    H::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},Nothing}
    Q::AbstractMatrix{Float64}
    F::Union{AbstractMatrix{Float64}, UniformScaling{Float64}}
end

struct ModelBuffer{T}
    x_new::AbstractVector{T}
    P_new::AbstractMatrix{T}
end

nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)

struct STARSInstrumentData
    data::AbstractArray
    bias::Union{Float64,AbstractArray} # scalar
    uq::Union{Float64,AbstractArray} # scalar
    dynamic_bias::Bool # true/false for if to model a dynamic bias term 
    dynamic_bias_coefs::Union{AbstractVector{Float64}, Nothing} # [ar coef, ar var]
    spatial_resolution::AbstractVector{Float64} # [rx,ry] vector of spatial resolution 
    dates::AbstractVector # vector of dates
    coords::AbstractArray # n x 2 array of spatial coordinates
end

### struct for instrument geospatial data
struct STARSInstrumentGeoData
    origin::AbstractVector # [rx,ry] vector of raster origin
    cell_size::AbstractVector # [rx,ry] vector of spatial resolution 
    ndims::AbstractVector # [nx,ny] vector of size of instrument grid
    fidelity::Int64 # [0,1,2] indicating 0: highest spatial res, 1: high spatial res, 2: coarse res
    dates::AbstractVector # vector of dates
end

"kalman filter one-step recursion"
function kalman_filter(M::KSModel{T}, B::ModelBuffer{T}, y::AbstractVector{T}, err_vars::AbstractVector{T}, x_pred::AbstractVector{T}, P_pred::AbstractMatrix{T}) where T <: Real

    # Shortcuts to use old syntax for now:
    Ht = @views M.H

    x_new = @views B.x_new
    P_new = @views B.P_new
    
    # res_pred = y - Ht * x_pred # innovation
    
    mul!(y, Ht, x_pred, -1.0, 1.0)
    
    HPpT = P_pred * Ht'

    S = Ht * HPpT .+ Diagonal(err_vars) # innovation covariance

    # Kalman gain; K = P_pred * H' * inv(S)
    begin
        LAPACK.potrf!('U', S)
        LAPACK.potri!('U', S)
        K = BLAS.symm('R', 'U', S, HPpT)
    end

    # With K
    x_new .= x_pred 
    mul!(x_new, K, y, 1.0, 1.0) # filtering distribution mean

    HP_pred = Ht * P_pred
    P_new .= P_pred
    mul!(P_new, K, HP_pred, -1.0, 1.0)

    return x_new, P_new
end

function kalman_filter!(x_new::AbstractVector{T}, P_new::AbstractMatrix{T}, 
        Ht::Union{AbstractSparseMatrix{T},AbstractMatrix{T}},
        y::AbstractVector{T}, err_vars::AbstractVector{T}, 
        x_pred::AbstractVector{T}, 
        P_pred::AbstractMatrix{T}) where T <: Real 

    mul!(y, Ht, x_pred, -1.0, 1.0)
    
    HPpT = P_pred * Ht'

    S = Ht * HPpT .+ Diagonal(err_vars) # innovation covariance

    # Kalman gain; K = P_pred * H' * inv(S)
    begin
        LAPACK.potrf!('U', S)
        LAPACK.potri!('U', S)
        K = BLAS.symm('R', 'U', S, HPpT)
    end

    # With K
    x_new .= x_pred 
    mul!(x_new, K, y, 1.0, 1.0) # filtering distribution mean

    HP_pred = Ht * P_pred
    P_new .= P_pred
    mul!(P_new, K, HP_pred, -1.0, 1.0);

    # return x_new, P_new
end

### add bias model components
function coarse_fine_fusion!(fused_image, 
        fused_sd_image,
        fused_bias_image,
        fused_bias_sd_image,
        measurements,
        target_coords,
        kp_ij,
        kp_bias_ij,
        prior_mean,
        prior_var,
        prior_bias_mean,
        prior_bias_var,
        model_pars;
        target_times = [1],
        spatial_mod::Function = mat32_cor,                                         
        obs_operator::Function = unif_weighted_obs_operator,
        state_in_cov::Bool = false,
        cov_wt::AbstractFloat = 0.3)
   
    ni = size(measurements)[1] 
    nf = size(target_coords)[1]
    nnobs = Vector{Int64}(undef, ni)
    t0v = Vector{Int64}(undef, ni)
    ttv = Vector{Int64}(undef, ni)
    for i in 1:ni
        nnobs[i] = size(measurements[i].data)[1]
        t0v[i] = measurements[i].dates[1]
        ttv[i] = measurements[i].dates[end] 
    end

    t0 = minimum(t0v)
    tt = maximum(ttv)
    tp = maximum(target_times);

    times = t0:tp

    nsteps = size(times)[1]

    K = 2;
    biases = [x.dynamic_bias for x in measurements]
    nn_biases = nnobs[:]
    nn_biases[.!biases] .= 0
    nt_bias = sum(nn_biases)

    data_kp = falses(ni,nsteps)

    ## build observation operator, stack observations and variances 
    Hl = Vector(undef,ni)
    Qb = Float64[]
    Fb = Float64[]

    for (i,x) in enumerate(measurements)
        Hs = obs_operator(x.coords, target_coords, x.spatial_resolution) # kwargs for uniform needs :target_resolution, # kwargs for gaussian needs :scale, :p
        if x.dynamic_bias 
            Hbb = [spzeros(x,x) for x in nn_biases]
            Hbb[i] .= 1.0*I(nnobs[i])
            Hl[i] = hcat(Hs,Hbb[length.(Hbb) .> 0]...)
            append!(Fb,x.dynamic_bias_coefs[1]*ones(nnobs[i]))
            append!(Qb,x.dynamic_bias_coefs[2]*ones(nnobs[i]))
        else
            Hl[i] = hcat(Hs,spzeros(nnobs[i],nt_bias))
        end
        data_kp[i,in(measurements[i].dates).(t0:tp)] .= 1
    end
    H = vcat(Hl...);

    nb = sum(length.(Qb))
    n = nf+nb
    Q = zeros(n,n)

    Q[1:nf,1:nf] = model_pars[1] .* spatial_mod(target_coords', model_pars[2:end]) 

    if length(Fb) .> 0
        F = Diagonal([ones(nf)...,Fb...])
        @views Q[diagind(Q)[(nf+1):end]] = [Qb...]
    else
        F = UniformScaling(1.0)
    end

    nb = length(prior_bias_mean)
    n = nf+nb

    filtering_means = zeros(n,nsteps+1)
    filtering_covs = zeros(n,n,nsteps+1)

    filtering_means[1:nf,1] = prior_mean
    filtering_means[(nf+1):end,1] = prior_bias_mean
    @views filtering_covs[diagind(filtering_covs[:,:,1])[1:nf]] = prior_var
    @views filtering_covs[diagind(filtering_covs[:,:,1])[(nf+1):end]] = prior_bias_var

    x_pred = zeros(n)
    P_pred = zeros(n,n)
    x_new = zeros(n)
    P_new = zeros(n,n)
    Qss = zeros(nf,nf)
    FPpred = similar(P_pred)

    tk = 1
    for (t,t2) in enumerate(t0:tp)
        if state_in_cov 
            Xtt = @views filtering_means[1:nf,t:t]
            pairwise!(Qss,Euclidean(1e-12), Xtt, dims=1) 
            phi = maximum([0.01,mean(Qss)])

            Qss ./= phi
            Qss .= exp.(-Qss) + UniformScaling(1e-8)
            # replace!(x->(x==1.0 ? 1.0+1e-8 : x), Qss) 
  
            Qss .*= model_pars[1] * (1.0 .- cov_wt) 
        
            @view(Q[1:nf,1:nf]) .*= cov_wt 
            @view(Q[1:nf,1:nf]) .+= Qss
        end

        if !any(data_kp[:,t])
            ys = fill(NaN,1)
            err_vars = fill(NaN,1)
            M = KSModel(nothing, Q, F)
        else
            ys = Float64[]
            err_vars = Float64[]
            yms = Int64[]
            nii = 0
            for x in 1:K
                yss = @views measurements[x].data[:,measurements[x].dates .== t2][:]
                ym = findall(.!isnan.(yss))
                if length(ym) > 0
                    err_varss = @views measurements[x].uq*ones(length(ym))
                    @views append!(ys,yss[ym]);
                    append!(err_vars,err_varss);
                    append!(yms,ym .+ nii);
                end
                nii += nnobs[x]
            end
            # M = KSModel(H[yms,:], Q, F)
            Ht = H[yms,:]
        end;

        # Predictive mean and covariance here
        P_pred .= Q
        mul!(x_pred, F, @view(filtering_means[:,t]))
        mul!(FPpred, F, @view(filtering_covs[:,:,t]))
        mul!(P_pred, FPpred, F', 1.0, 1.0)

        # Filtering is done here
        if sum(.!isnan.(ys)) == 0
            filtering_means[:,t+1] = x_pred
            filtering_covs[:,:,t+1] = P_pred
        else
            kalman_filter!(x_new, P_new, Ht, ys, err_vars, x_pred, P_pred)
            filtering_means[:,t+1] = x_new
            filtering_covs[:,:,t+1] = P_new
        end    

        nbau = size(kp_ij,1)

        if t2 .∈ Ref(target_times)
            fused_image[kp_ij,tk] = @views filtering_means[1:nbau,t+1];
            fused_sd_image[kp_ij,tk] = @views diag(filtering_covs[1:nbau,1:nbau,t+1])
                    
            fused_bias_image[kp_bias_ij,tk] = @views filtering_means[nf+1,t+1]
            fused_bias_sd_image[kp_bias_ij,tk] = @views filtering_covs[nf+1,nf+1,t+1]
            tk += 1
        end
    end    
end

"function to perform data fusion across scene from multiple instruments"
function coarse_fine_scene_fusion(fine_data, coarse_data,
        fine_geodata, coarse_geodata,
        nwindows::AbstractVector,
        prior_mean::AbstractArray,
        prior_var::AbstractArray,
        prior_bias_mean::AbstractArray,
        prior_bias_var::AbstractArray,
        model_pars::AbstractArray;
        nsamp = 100,
        window_buffer = 2,
        target_times = [1], 
        spatial_mod::Function = matern_cor,                                           
        obs_operator::Function = unif_weighted_obs_operator,
        state_in_cov = true,
        cov_wt = 0.2,
        nb_coarse=2.0,
        show_progress_bar::Bool=true) 

    ### define target extent and target + buffer extent
    window_csize = coarse_geodata.cell_size
    target_csize = fine_geodata.cell_size
    window_origin = coarse_geodata.origin
    window_ndims = coarse_geodata.ndims
    target_origin = fine_geodata.origin
    target_ndims = fine_geodata.ndims

    K = 2
    tkp = size(target_times,1)
    fused_image = zeros(target_ndims[1], target_ndims[2], tkp);
    fused_sd_image = zeros(target_ndims[1], target_ndims[2], tkp);

    fused_bias_image = zeros(window_ndims[1], window_ndims[2], tkp);
    fused_bias_sd_image = zeros(window_ndims[1], window_ndims[2], tkp);

    inds = hcat(repeat(1:nwindows[1], inner=nwindows[2]), repeat(1:nwindows[2], outer=nwindows[1]))

    inst_geodata = [fine_geodata, coarse_geodata]

    n = size(inds,1)
    if show_progress_bar
        p = Progress(n)
        update!(p, 0)
        jj = Threads.Atomic{Int}(0)
        j = Threads.SpinLock()
    end

    Threads.@threads for ii in 1:n
        if show_progress_bar
            Threads.atomic_add!(jj, 1)
            Threads.lock(j)
            update!(p, jj[])
            Threads.unlock(j)
        end

        k,l = inds[ii,:]
        ### find target partition given origin and (k,l)th partition coordinate
        bbox_centroid = window_origin .+ [k-1, l-1].*window_csize
        window_bbox = bbox_from_centroid(bbox_centroid, window_csize)

        ### add buffer of window_buffer target pixels around target partition extent
        buffer_ext = window_bbox .+ window_buffer*[-1.01,1.01]*target_csize'

        ### find extent of overlapping instruments for each instrument
        all_exts = Vector{AbstractMatrix{Float64}}(undef,K)
        for (i,x) in enumerate(inst_geodata)
            all_exts[i] = Matrix(find_overlapping_ext(buffer_ext[1,:], buffer_ext[2,:], x.origin, x.cell_size))
        end

        ## extend window to number of coarse neighbors
        exx = window_bbox .+ [-nb_coarse - 0.01,nb_coarse + 0.01]*inst_geodata[2].cell_size'
        push!(all_exts, exx)

        ### finf full extent combining all instrument extents
        full_ext = merge_extents(all_exts, sign.(target_csize))

        ### Find all BAUs within target
        target_ij = find_all_ij_ext(window_bbox[1,:], window_bbox[2,:], target_origin, target_csize, target_ndims; inclusive=false)
        # t_xy = get_sij_from_ij(target_ij, target_origin, target_csize)

        ### Find all BAUs within target + buffer
        ss_target = find_all_ij_ext(buffer_ext[1,:], buffer_ext[2,:], target_origin, target_csize, target_ndims)
        # tb_xy = get_sij_from_ij(target_buffer_ij, target_origin, target_csize)

        ### subsample BAUs within full extent of coarse pixels
        ss_samp = sobol_bau_ij(full_ext[1,:], full_ext[2,:], target_origin, target_csize, target_ndims; nsamp=nsamp)
        bau_ij = unique(vcat(target_ij, ss_target, ss_samp),dims=1)
        bau_ci = CartesianIndex.(bau_ij[:,1],bau_ij[:,2])

        bau_xy = get_sij_from_ij(bau_ij, target_origin, target_csize)

        ### Find measurements:
        measurements = Vector{STARSInstrumentData}(undef, K)

        # fine_ij = unique(find_nearest_ij_multi(ss_xy, fine_geodata.origin, fine_geodata.cell_size,fine_geodata.ndims),dims=1)
        # fine_xy = get_sij_from_ij(fine_ij, fine_geodata.origin, fine_geodata.cell_size)
        # ys = fine_data.data[bau_ci,:]
        measurements[1] = @views STARSInstrumentData(fine_data.data[bau_ci,:],
                                fine_data.bias, 
                                fine_data.uq, 
                                fine_data.dynamic_bias,
                                fine_data.dynamic_bias_coefs,
                                abs.(fine_geodata.cell_size),
                                fine_geodata.dates,
                                bau_xy)

        coarse_ij = find_all_ij_ext(full_ext[1,:], full_ext[2,:], coarse_geodata.origin, coarse_geodata.cell_size, coarse_geodata.ndims; inclusive=false)
        coarse_xy = get_sij_from_ij(coarse_ij, coarse_geodata.origin, coarse_geodata.cell_size)

        kp_bias = findall(sum(abs.(coarse_xy .- bbox_centroid'),dims=2)[:] .== 0)[1]
        pp = [kp_bias, 1:(kp_bias-1)..., (kp_bias+1):size(coarse_ij,1)...]

        coarse_ci = CartesianIndex.(coarse_ij[pp,1],coarse_ij[pp,2])
        # ys = coarse_data.data[coarse_ci,:]
        measurements[2] = @views STARSInstrumentData(coarse_data.data[coarse_ci,:],
                            coarse_data.bias, 
                            coarse_data.uq, 
                            coarse_data.dynamic_bias,
                            coarse_data.dynamic_bias_coefs,
                            abs.(coarse_geodata.cell_size),
                            coarse_geodata.dates,
                            coarse_xy[pp,:])

        ### x,y coords for all baus
        bau_coords = get_sij_from_ij(bau_ij, target_origin, target_csize)

        ### subset prior mean and var arrays to bau pixels
        prior_mean_sub = @views prior_mean[bau_ci][:]
        prior_var_sub = @views prior_var[bau_ci][:]

        prior_bias_mean_sub = @views prior_bias_mean[coarse_ci][:]
        prior_bias_var_sub = @views prior_bias_var[coarse_ci][:]

        #### order bias?
        t_ind = CartesianIndex.(target_ij[:,1], target_ij[:,2])
        b_ind = CartesianIndex(coarse_ij[kp_bias,1],coarse_ij[kp_bias,2])

        model_pars_sub = model_pars[b_ind,:]
        
        coarse_fine_fusion!(fused_image, 
                fused_sd_image,
                fused_bias_image,
                fused_bias_sd_image,
                measurements,
                bau_coords,
                t_ind,
                b_ind,
                prior_mean_sub,
                prior_var_sub,
                prior_bias_mean_sub,
                prior_bias_var_sub,
                model_pars_sub;
                target_times = target_times,
                spatial_mod = spatial_mod,                                         
                obs_operator = obs_operator,
                state_in_cov = state_in_cov,
                cov_wt = cov_wt)
    end
    return fused_image, fused_sd_image, fused_bias_image, fused_bias_sd_image
end

function coarse_fine_fusion_ns!(fused_image, 
        fused_sd_image,
        fused_bias_image,
        fused_bias_sd_image,
        measurements,
        target_coords,
        kp_ij,
        kp_bias_ij,
        prior_mean,
        prior_var,
        prior_bias_mean,
        prior_bias_var,
        model_pars; ## at fine resolution
        target_times = [1],
        spatial_mod::Function = mat32_cor,                                         
        obs_operator::Function = unif_weighted_obs_operator,
        state_in_cov::Bool = false,
        cov_wt::AbstractFloat = 0.3)
   
    ni = size(measurements)[1] 
    nf = size(target_coords)[1]
    nnobs = Vector{Int64}(undef, ni)
    t0v = Vector{Int64}(undef, ni)
    ttv = Vector{Int64}(undef, ni)
    for i in 1:ni
        nnobs[i] = size(measurements[i].data)[1]
        t0v[i] = measurements[i].dates[1]
        ttv[i] = measurements[i].dates[end] 
    end

    t0 = minimum(t0v)
    tt = maximum(ttv)
    tp = maximum(target_times);

    times = t0:tp

    nsteps = size(times)[1]

    K = 2;
    biases = [x.dynamic_bias for x in measurements]
    nn_biases = nnobs[:]
    nn_biases[.!biases] .= 0
    nt_bias = sum(nn_biases)

    data_kp = falses(ni,nsteps)

    ## build observation operator, stack observations and variances 
    Hl = Vector(undef,ni)
    Qb = Float64[]
    Fb = Float64[]

    for (i,x) in enumerate(measurements)
        Hs = obs_operator(x.coords, target_coords, x.spatial_resolution) # kwargs for uniform needs :target_resolution, # kwargs for gaussian needs :scale, :p
        if x.dynamic_bias 
            Hbb = [spzeros(x,x) for x in nn_biases]
            Hbb[i] .= 1.0*I(nnobs[i])
            Hl[i] = hcat(Hs,Hbb[length.(Hbb) .> 0]...)
            append!(Fb,x.dynamic_bias_coefs[1]*ones(nnobs[i]))
            append!(Qb,x.dynamic_bias_coefs[2]*ones(nnobs[i]))
        else
            Hl[i] = hcat(Hs,spzeros(nnobs[i],nt_bias))
        end
        data_kp[i,in(measurements[i].dates).(t0:tp)] .= 1
    end
    H = vcat(Hl...);

    nb = sum(length.(Qb))
    n = nf+nb
    Q = zeros(n,n)
    cvs = Diagonal(model_pars[:,1]) ## sqrt of variances
    Q[1:nf,1:nf] = cvs * spatial_mod(target_coords', model_pars[1,2:end]) * cvs

    if length(Fb) .> 0
        F = Diagonal([ones(nf)...,Fb...])
        @views Q[diagind(Q)[(nf+1):end]] = [Qb...]
    else
        F = UniformScaling(1.0)
    end

    nb = length(prior_bias_mean)
    n = nf+nb

    filtering_means = zeros(n,nsteps+1)
    filtering_covs = zeros(n,n,nsteps+1)

    filtering_means[1:nf,1] = prior_mean
    filtering_means[(nf+1):end,1] = prior_bias_mean
    @views filtering_covs[diagind(filtering_covs[:,:,1])[1:nf]] = prior_var
    @views filtering_covs[diagind(filtering_covs[:,:,1])[(nf+1):end]] = prior_bias_var

    x_pred = zeros(n)
    P_pred = zeros(n,n)
    x_new = zeros(n)
    P_new = zeros(n,n)
    Qss = zeros(nf,nf)
    FPpred = similar(P_pred)

    tk = 1
    for (t,t2) in enumerate(t0:tp)
        if state_in_cov 
            Xtt = @views filtering_means[1:nf,1:t]
            pairwise!(Qss,Euclidean(1e-12), Xtt, dims=1) 
            phi = maximum([0.01,mean(Qss)])

            Qss ./= phi
            Qss .= cvs * exp.(-Qss) * cvs + UniformScaling(1e-8)
            # replace!(x->(x==1.0 ? 1.0+1e-8 : x), Qss) 
  
            Qss .*= (1.0 .- cov_wt) 
        
            @view(Q[1:nf,1:nf]) .*= cov_wt 
            @view(Q[1:nf,1:nf]) .+= Qss
        end

        if !any(data_kp[:,t])
            ys = fill(NaN,1)
            err_vars = fill(NaN,1)
            M = KSModel(nothing, Q, F)
        else
            ys = Float64[]
            err_vars = Float64[]
            yms = Int64[]
            nii = 0
            for x in 1:K
                yss = @views measurements[x].data[:,measurements[x].dates .== t2][:]
                ym = findall(.!isnan.(yss))
                if length(ym) > 0
                    err_varss = @views measurements[x].uq*ones(length(ym))
                    @views append!(ys,yss[ym]);
                    append!(err_vars,err_varss);
                    append!(yms,ym .+ nii);
                end
                nii += nnobs[x]
            end
            # M = KSModel(H[yms,:], Q, F)
            Ht = H[yms,:]
        end;

        # Predictive mean and covariance here
        P_pred .= Q
        mul!(x_pred, F, @view(filtering_means[:,t]))
        mul!(FPpred, F, @view(filtering_covs[:,:,t]))
        mul!(P_pred, FPpred, F', 1.0, 1.0)

        # Filtering is done here
        if sum(.!isnan.(ys)) == 0
            filtering_means[:,t+1] = x_pred
            filtering_covs[:,:,t+1] = P_pred
        else
            kalman_filter!(x_new, P_new, Ht, ys, err_vars, x_pred, P_pred)
            filtering_means[:,t+1] = x_new
            filtering_covs[:,:,t+1] = P_new
        end    

        nbau = size(kp_ij,1)

        if t2 .∈ Ref(target_times)
            fused_image[kp_ij,tk] = @views filtering_means[1:nbau,t+1];
            fused_sd_image[kp_ij,tk] = @views diag(filtering_covs[1:nbau,1:nbau,t+1])
                    
            fused_bias_image[kp_bias_ij,tk] = @views filtering_means[nf+1,t+1]
            fused_bias_sd_image[kp_bias_ij,tk] = @views filtering_covs[nf+1,nf+1,t+1]
            tk += 1
        end
    end    
end

function coarse_fine_scene_fusion_ns(fine_data, coarse_data,
        fine_geodata, coarse_geodata,
        nwindows::AbstractVector,
        prior_mean::AbstractArray,
        prior_var::AbstractArray,
        prior_bias_mean::AbstractArray,
        prior_bias_var::AbstractArray,
        model_pars::AbstractArray;
        nsamp = 100,
        window_buffer = 2,
        target_times = [1], 
        spatial_mod::Function = matern_cor,                                           
        obs_operator::Function = unif_weighted_obs_operator,
        state_in_cov = true,
        cov_wt = 0.2,
        nb_coarse=2.0,
        show_progress_bar::Bool=true) 

    ### define target extent and target + buffer extent
    window_csize = coarse_geodata.cell_size
    target_csize = fine_geodata.cell_size
    window_origin = coarse_geodata.origin
    window_ndims = coarse_geodata.ndims
    target_origin = fine_geodata.origin
    target_ndims = fine_geodata.ndims

    K = 2
    tkp = size(target_times,1)
    fused_image = zeros(target_ndims[1], target_ndims[2], tkp);
    fused_sd_image = zeros(target_ndims[1], target_ndims[2], tkp);

    fused_bias_image = zeros(window_ndims[1], window_ndims[2], tkp);
    fused_bias_sd_image = zeros(window_ndims[1], window_ndims[2], tkp);

    inds = hcat(repeat(1:nwindows[1], inner=nwindows[2]), repeat(1:nwindows[2], outer=nwindows[1]))

    inst_geodata = [fine_geodata, coarse_geodata]

    n = size(inds,1)
    if show_progress_bar
        p = Progress(n)
        update!(p, 0)
        jj = Threads.Atomic{Int}(0)
        j = Threads.SpinLock()
    end

    Threads.@threads for ii in 1:n
        if show_progress_bar
            Threads.atomic_add!(jj, 1)
            Threads.lock(j)
            update!(p, jj[])
            Threads.unlock(j)
        end
    # for ii in 1:n
        k,l = inds[ii,:]
        ### find target partition given origin and (k,l)th partition coordinate
        bbox_centroid = window_origin .+ [k-1, l-1].*window_csize
        window_bbox = bbox_from_centroid(bbox_centroid, window_csize)

        ### add buffer of window_buffer target pixels around target partition extent
        buffer_ext = window_bbox .+ window_buffer*[-1.01,1.01]*target_csize'

        ### find extent of overlapping instruments for each instrument
        all_exts = Vector{AbstractMatrix{Float64}}(undef,K)
        for (i,x) in enumerate(inst_geodata)
            all_exts[i] = Matrix(find_overlapping_ext(buffer_ext[1,:], buffer_ext[2,:], x.origin, x.cell_size))
        end

        ## extend window to number of coarse neighbors
        exx = window_bbox .+ [-nb_coarse - 0.01,nb_coarse + 0.01]*inst_geodata[2].cell_size'
        push!(all_exts, exx)

        ### finf full extent combining all instrument extents
        full_ext = merge_extents(all_exts, sign.(target_csize))

        ### Find all BAUs within target
        target_ij = find_all_ij_ext(window_bbox[1,:], window_bbox[2,:], target_origin, target_csize, target_ndims; inclusive=false)
        # t_xy = get_sij_from_ij(target_ij, target_origin, target_csize)

        ### Find all BAUs within target + buffer
        ss_target = find_all_ij_ext(buffer_ext[1,:], buffer_ext[2,:], target_origin, target_csize, target_ndims)
        # tb_xy = get_sij_from_ij(target_buffer_ij, target_origin, target_csize)

        ### subsample BAUs within full extent of coarse pixels
        ss_samp = sobol_bau_ij(full_ext[1,:], full_ext[2,:], target_origin, target_csize, target_ndims; nsamp=nsamp)
        bau_ij = unique(vcat(target_ij, ss_target, ss_samp),dims=1)
        bau_ci = CartesianIndex.(bau_ij[:,1],bau_ij[:,2])

        bau_xy = get_sij_from_ij(bau_ij, target_origin, target_csize)

        ### Find measurements:
        measurements = Vector{STARSInstrumentData}(undef, K)

        # fine_ij = unique(find_nearest_ij_multi(ss_xy, fine_geodata.origin, fine_geodata.cell_size,fine_geodata.ndims),dims=1)
        # fine_xy = get_sij_from_ij(fine_ij, fine_geodata.origin, fine_geodata.cell_size)
        # ys = fine_data.data[bau_ci,:]
        measurements[1] = @views STARSInstrumentData(fine_data.data[bau_ci,:],
                                fine_data.bias, 
                                fine_data.uq, 
                                fine_data.dynamic_bias,
                                fine_data.dynamic_bias_coefs,
                                abs.(fine_geodata.cell_size),
                                fine_geodata.dates,
                                bau_xy)

        coarse_ij = find_all_ij_ext(full_ext[1,:], full_ext[2,:], coarse_geodata.origin, coarse_geodata.cell_size, coarse_geodata.ndims; inclusive=false)
        coarse_xy = get_sij_from_ij(coarse_ij, coarse_geodata.origin, coarse_geodata.cell_size)

        kp_bias = findall(sum(abs.(coarse_xy .- bbox_centroid'),dims=2)[:] .== 0)[1]
        pp = [kp_bias, 1:(kp_bias-1)..., (kp_bias+1):size(coarse_ij,1)...]

        coarse_ci = CartesianIndex.(coarse_ij[pp,1],coarse_ij[pp,2])
        # ys = coarse_data.data[coarse_ci,:]
        measurements[2] = @views STARSInstrumentData(coarse_data.data[coarse_ci,:],
                            coarse_data.bias, 
                            coarse_data.uq, 
                            coarse_data.dynamic_bias,
                            coarse_data.dynamic_bias_coefs,
                            abs.(coarse_geodata.cell_size),
                            coarse_geodata.dates,
                            coarse_xy[pp,:])

        ### x,y coords for all baus
        bau_coords = get_sij_from_ij(bau_ij, target_origin, target_csize)

        ### subset prior mean and var arrays to bau pixels
        prior_mean_sub = @views prior_mean[bau_ci][:]
        prior_var_sub = @views prior_var[bau_ci][:]

        prior_bias_mean_sub = @views prior_bias_mean[coarse_ci][:]
        prior_bias_var_sub = @views prior_bias_var[coarse_ci][:]

        #### order bias?
        t_ind = CartesianIndex.(target_ij[:,1], target_ij[:,2])
        b_ind = CartesianIndex(coarse_ij[kp_bias,1],coarse_ij[kp_bias,2])

        model_pars_sub = model_pars[bau_ci,:]

        coarse_fine_fusion_ns!(fused_image, 
                fused_sd_image,
                fused_bias_image,
                fused_bias_sd_image,
                measurements,
                bau_coords,
                t_ind,
                b_ind,
                prior_mean_sub,
                prior_var_sub,
                prior_bias_mean_sub,
                prior_bias_var_sub,
                model_pars_sub;
                target_times = target_times,
                spatial_mod = spatial_mod,                                         
                obs_operator = obs_operator,
                state_in_cov = state_in_cov,
                cov_wt = cov_wt)
    end
    return fused_image, fused_sd_image, fused_bias_image, fused_bias_sd_image
end
# end

