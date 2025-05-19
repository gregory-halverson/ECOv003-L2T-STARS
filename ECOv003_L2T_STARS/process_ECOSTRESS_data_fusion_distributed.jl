using Glob
using Dates
using Rasters
using LinearAlgebra
using STARSDataFusion
using STARSDataFusion.BBoxes
using STARSDataFusion.sentinel_tiles
using STARSDataFusion.HLS
using STARSDataFusion.VNP43
using Logging
using Pkg
using Statistics
using Distributed
Pkg.add("OpenSSL")
using HTTP
using JSON

function read_json(file::String)::Dict
    open(file, "r") do f
        return JSON.parse(f)
    end
end

function write_json(file::String, data::Dict)
    open(file, "w") do f
        JSON.print(f, data)
    end
end

@info "processing STARS data fusion"

# wrkrs = parse(Int64, ARGS[1])
wrkrs = 8

@info "starting $(wrkrs) workers"
addprocs(wrkrs) ## need to set num workers

@everywhere using STARSDataFusion
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

DEFAULT_MEAN = 0.12
DEFAULT_SD = 0.01

#### add bias components
## need to read-in previous bias value with prior
DEFAULT_BIAS = 0.0

struct CustomLogger <: AbstractLogger
    stream::IO
    min_level::LogLevel
end

Logging.min_enabled_level(logger::CustomLogger) = logger.min_level

function Logging.shouldlog(logger::CustomLogger, level, _module, group, id)
    return level >= logger.min_level
end

function Logging.handle_message(logger::CustomLogger, level, message, _module, group, id, file, line; kwargs...)
    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println(logger.stream, "[$t $(uppercase(string(level)))] $message")
end

global_logger(CustomLogger(stdout, Logging.Info))

# command = f'cd "{STARS_source_directory}" && julia --project=@. "{julia_script_filename}" "{tile}" "{coarse_cell_size}" "{fine_cell_size}" "{VIIRS_start_date}" "{VIIRS_end_date}" "{HLS_start_date}" "{HLS_end_date}" "{coarse_directory}" "{fine_directory}" "{posterior_filename}" "{posterior_UQ_filename}" "{posterior_bias_filename}" "{posterior_bias_UQ_filename}" "{prior_filename}" "{prior_UQ_filename}" "{prior_bias_filename}" "{prior_bias_UQ_filename}"'

tile = ARGS[2]
@info "tile: $(tile)"
coarse_cell_size = parse(Int64, ARGS[3])
@info "coarse cell size: $(coarse_cell_size)"
fine_cell_size = parse(Int64, ARGS[4])
@info "fine cell size: $(fine_cell_size)"
VIIRS_start_date = Date(ARGS[5])
@info "VIIRS start date: $(VIIRS_start_date)"
VIIRS_end_date = Date(ARGS[6])
@info "VIIRS end date: $(VIIRS_end_date)"
HLS_start_date = Date(ARGS[7])
@info "HLS start date: $(HLS_start_date)"
HLS_end_date = Date(ARGS[8])
@info "HLS end date: $(HLS_end_date)"
coarse_directory = ARGS[9]
@info "coarse inputs directory: $(coarse_directory)"
fine_directory = ARGS[10]
@info "fine inputs directory: $(fine_directory)"
posterior_filename = ARGS[11]
@info "posterior filename: $(posterior_filename)"
posterior_UQ_filename = ARGS[12]
@info "posterior UQ filename: $(posterior_UQ_filename)"
posterior_flag_filename = ARGS[13]
@info "posterior flag filename: $(posterior_flag_filename)"
posterior_bias_filename = ARGS[14]
@info "posterior bias filename: $(posterior_bias_filename)"

if size(ARGS)[1] >= 18
    prior_filename = ARGS[15]
    @info "prior filename: $(prior_filename)"
    prior_mean = Array(Raster(prior_filename))
    prior_UQ_filename = ARGS[16]
    @info "prior UQ filename: $(prior_UQ_filename)"
    prior_sd = Array(Raster(prior_UQ_filename))
    prior_flag_filename = ARGS[17]
    @info "prior flag filename: $(prior_flag_filename)"
    prior_flag = Array(Raster(prior_flag_filename))
    prior_bias_filename = ARGS[18]
    @info "prior bias filename: $(prior_bias_filename)"
    prior_bias_factor = Float64(read_json(prior_bias_filename)["coarse_bias"])
else
    prior_mean = nothing
    prior_bias_factor = DEFAULT_BIAS
end

x_coarse, y_coarse = sentinel_tile_dims(tile, coarse_cell_size)
x_coarse_size = size(x_coarse)[1]
y_coarse_size = size(y_coarse)[1]
@info "coarse x size: $(x_coarse_size)"
@info "coarse y size: $(y_coarse_size)"
x_fine, y_fine = sentinel_tile_dims(tile, fine_cell_size)
x_fine_size = size(x_fine)[1]
y_fine_size = size(y_fine)[1]
@info "fine x size: $(x_fine_size)"
@info "fine y size: $(y_fine_size)"

coarse_image_filenames = sort(glob("*.tif", coarse_directory))
coarse_dates_found = [Date(split(basename(filename), "_")[3]) for filename in coarse_image_filenames]

fine_image_filenames = sort(glob("*.tif", fine_directory))
fine_dates_found = [Date(split(basename(filename), "_")[3]) for filename in fine_image_filenames]

coarse_start_date = VIIRS_start_date
coarse_end_date = VIIRS_end_date

fine_start_date = HLS_start_date
fine_end_date = HLS_end_date

dates = [fine_start_date + Day(d - 1) for d in 1:((fine_end_date - fine_start_date).value + 1)]
t = Ti(dates)
coarse_dims = (x_coarse, y_coarse, t)
fine_dims = (x_fine, y_fine, t)

covariance_dates = [coarse_start_date + Day(d - 1) for d in 1:((coarse_end_date - coarse_start_date).value + 1)]
t_covariance = Ti(covariance_dates)
covariance_dims = (x_coarse, y_coarse, t_covariance)

covariance_images = []

for (i, date) in enumerate(covariance_dates)
    date = Dates.format(date, dateformat"yyyy-mm-dd")
    match = findfirst(x -> occursin(date, x), coarse_image_filenames)
    timestep_index = Band(i:i)
    timestep_dims = (x_coarse, y_coarse, timestep_index)

    if match === nothing
        @info "coarse image is not available on $(date)"
        covariance_image = Raster(fill(NaN, x_coarse_size, y_coarse_size, 1), dims=timestep_dims, missingval=NaN)
        @info size(covariance_image)
    else
        filename = coarse_image_filenames[match]
        @info "ingesting coarse image on $(date): $(filename)"
        covariance_image = Raster(reshape(Raster(filename), x_coarse_size, y_coarse_size, 1), dims=timestep_dims, missingval=NaN)
        replace!(covariance_image, missing => NaN)
        @info size(covariance_image)
    end

    push!(covariance_images, covariance_image)
end

@info "concatenating coarse images for covariance calculation"
covariance_images = Raster(cat(covariance_images..., dims=3), dims=covariance_dims, missingval=NaN)

# estimate spatial var parameter
n_eff = compute_n_eff(Int(round(coarse_cell_size / fine_cell_size)), 2, smoothness=1.5) ## Matern: range = 200m, smoothness = 1.5
sp_var = fast_var_est(covariance_images, n_eff_agg = n_eff)
# cov_pars_raster = Raster(fill(NaN, size(covariance_images)[1], size(covariance_images)[2], 4), dims=(covariance_images.dims[1:2]...,Band(1:4)), missingval=covariance_images.missingval)
# cov_pars_raster[:,:,1] = sp_var
# cov_pars_raster[:,:,2] .= 200
# cov_pars_raster[:,:,3] .= 1e-10
# cov_pars_raster[:,:,4] .= 1.5

# cov_pars = ones((size(covariance_images)[1], size(covariance_images)[2], 4))
# cov_pars[:,:,1] = Array{Float64}(sp_var)
# cov_pars[:,:,2] .= 200
# cov_pars[:,:,3] .= 1e-10
# cov_pars[:,:,4] .= 1.5

coarse_images = []
coarse_dates = Vector{Date}(undef,0)

tk=1
for (i, date) in enumerate(dates)
    date = Dates.format(date, dateformat"yyyy-mm-dd")
    matched = findfirst(x -> occursin(date, x), coarse_image_filenames)

    if matched === nothing
        @info "coarse image is not available on $(date)"
        # coarse_image = Raster(fill(NaN, x_coarse_size, y_coarse_size, 1), dims=timestep_dims, missingval=NaN)
        # @info size(coarse_image)
    else
        timestep_index = Band(tk:tk)
        timestep_dims = (x_coarse, y_coarse, timestep_index)
        filename = coarse_image_filenames[matched]
        @info "ingesting coarse image on $(date): $(filename)"
        coarse_image = Raster(reshape(Raster(filename), x_coarse_size, y_coarse_size, 1), dims=timestep_dims, missingval=NaN)
        replace!(coarse_image, missing => NaN)
        @info size(coarse_image)
        push!(coarse_dates, dates[i])
        push!(coarse_images, coarse_image)
        global tk += 1
    end
end
@info "concatenating coarse image inputs"
if length(coarse_images) == 0
    coarse_images = Raster(fill(NaN, x_coarse_size, y_coarse_size, 1), dims=(coarse_dims[1:2]..., Band(1:1)), missingval=NaN)
    coarse_array = zeros(x_coarse_size, y_coarse_size, 1)
    coarse_array .= NaN
    coarse_dates = [dates[1]]
else
    coarse_images = Raster(cat(coarse_images..., dims=3), dims=(coarse_dims[1:2]..., Band(1:length(coarse_dates))), missingval=NaN)
    coarse_array = Array{Float64}(coarse_images)
end

fine_images = []
fine_dates = Vector{Date}(undef,0)

tk=1
for (i, date) in enumerate(dates)
    date = Dates.format(date, dateformat"yyyy-mm-dd")
    match = findfirst(x -> occursin(date, x), fine_image_filenames)

    if match === nothing
        @info "fine image is not available on $(date)"
        # fine_image = Raster(fill(NaN, x_fine_size, y_fine_size, 1), dims=timestep_dims, missingval=NaN)
        # @info size(fine_image)
    else
        timestep_index = Band(tk:tk)
        timestep_dims = (x_fine, y_fine, timestep_index)
        filename = fine_image_filenames[match]
        @info "ingesting fine image on $(date): $(filename)"
        fine_image = Raster(reshape(Raster(filename), x_fine_size, y_fine_size, 1), dims=timestep_dims, missingval=NaN)
        replace!(fine_image, missing => NaN)
        @info size(fine_image)
        push!(fine_images, fine_image)
        push!(fine_dates, dates[i])
        global tk += 1
    end
end

@info "concatenating fine image inputs"
if length(fine_images) == 0
    fine_images = Raster(fill(NaN, x_fine_size, y_fine_size, 1), dims=(fine_dims[1:2]..., Band(1:1)), missingval=NaN)
    fine_array = zeros(x_fine_size, y_fine_size, 1)
    fine_array .= NaN
    fine_dates = [dates[1]]
else
    fine_images = Raster(cat(fine_images..., dims=3), dims=(fine_dims[1:2]..., Band(1:length(fine_dates))), missingval=NaN)
    fine_array = Array{Float64}(fine_images)
end

#### scene-level additive bias correction
common_dates = intersect(coarse_dates, fine_dates)

coarse_bias = 0
### don't bias correct NDVI
if occursin("albedo", posterior_filename)
    if length(common_dates) > 0 
        kpf = fine_dates .∈ Ref(common_dates)
        kpc = coarse_dates .∈ Ref(common_dates)

        coarse_images2 = @views coarse_images[:,:,kpc]
        fine_images2 = @views fine_images[:,:,kpf]

        fine_images_cs = resample(fine_images2, to = coarse_images2[:,:,1], method=:average)

        errs = Array(coarse_images2) .- Array(fine_images_cs)

        prop_scene = mean(sum(.!isnan.(errs), dims=3) .> 0)

        if prop_scene > 0.25 # must observe 25% of scene to estimate bias
            coarse_bias = nanmean(errs[fine_images_cs .> 0]) ## for NDVI, estimate bias from land
        else
            coarse_bias = prior_bias_factor
        end
    else
        coarse_bias = prior_bias_factor
    end
    coarse_array .-= coarse_bias ### correct VIIRS to HLS
    ## or 
    # fine_array .+= coarse_bias ### correct HLS to VIIRS
end

target_date = dates[end]
target_time = length(dates)

## 0, 1 mask
if isnothing(prior_mean)
    prior_flag = 2 .* ones(size(fine_images)[1:2])
    fine_pixels = sum(.!isnan.(fine_images),dims=3) 
    ## uncomment to keep viirs-only pixels 
    if sum(fine_pixels.==0) > 0
        coarse_nans = resample(sum(.!isnan.(coarse_images),dims=3), to=fine_images[:,:,1], method=:near)
        prior_flag[coarse_nans[:,:,1] .> 0] .= 1
    end

    prior_flag[fine_pixels[:,:,1] .> 0] .= 0

elseif sum(prior_flag .> 0) > 0
    fine_pixels = sum(.!isnan.(fine_images),dims=3)  
    prior_flag[fine_pixels[:,:,1] .> 0] .= 0

    nohls = prior_flag .> 0 
    ### uncomment to keep viirs-only pixels 
    if sum(nohls) > 0
        coarse_nans = resample(sum(.!isnan.(coarse_images),dims=3), to=fine_images[:,:,1], method=:near)
        prior_flag[(coarse_nans[:,:,1] .> 0) .& (nohls .== 1)] .= 1
    end

else
    prior_flag = zeros(size(fine_images)[1:2])
end

@info "running data fusion"

#### new approach
fine_times = findall(dates .∈ Ref(fine_dates))
coarse_times = findall(dates .∈ Ref(coarse_dates))

fine_ndims = collect(size(fine_images)[1:2])
coarse_ndims = collect(size(coarse_images)[1:2])

## instrument origins and cell sizes
fine_origin = get_centroid_origin_raster(fine_images)
coarse_origin = get_centroid_origin_raster(coarse_images)

fine_csize = collect(cell_size(fine_images))
coarse_csize = collect(cell_size(coarse_images))

fine_geodata = STARSInstrumentGeoData(fine_origin, fine_csize, fine_ndims, 0, fine_times)
coarse_geodata = STARSInstrumentGeoData(coarse_origin, coarse_csize, coarse_ndims, 2, coarse_times)
window_geodata = STARSInstrumentGeoData(coarse_origin, coarse_csize, coarse_ndims, 0, coarse_times)
# target_geodata = STARSInstrumentGeoData(fine_origin, fine_csize, fine_ndims, 0, 1:length(dates))

scf = 4
nwindows = Int.(ceil.(fine_ndims./scf))
target_ndims = nwindows.*scf
window_geodata = STARSInstrumentGeoData(fine_origin .+ (scf-1.0)/2.0*fine_csize, scf*fine_csize, nwindows, 0, 1:target_time)

fine_data = STARSInstrumentData(fine_array, 0.0, 1e-6, false, nothing, abs.(fine_csize), fine_times, [1. 1.])
coarse_data = STARSInstrumentData(coarse_array, 0.0, 1e-4, false, nothing, abs.(coarse_csize), coarse_times, [1. 1.])

nsamp=60
window_buffer = 5 ## set these differently for NDVI and albedo?

cov_pars = ones((size(fine_images)[1], size(fine_images)[2], 4))

sp_rs = resample(log.(sqrt.(sp_var[:,:,1])); to=fine_images[:,:,1], size=size(fine_images)[1:2], method=:cubicspline)
sp_rs[isnan.(sp_rs)] .= nanmean(sp_rs) ### the resampling won't go outside extent

cov_pars[:,:,1] = Array{Float64}(exp.(sp_rs))
cov_pars[:,:,2] .= coarse_cell_size
# cov_pars[:,:,2] .= 200.0
cov_pars[:,:,3] .= 1e-10
cov_pars[:,:,4] .= 0.5

if isnothing(prior_mean)
    fused_images, fused_sd_images = coarse_fine_scene_fusion_pmap(fine_data,
        coarse_data,
        fine_geodata, 
        coarse_geodata,
        window_geodata,
        DEFAULT_MEAN .* ones(fine_ndims...),
        DEFAULT_SD^2 .* ones(fine_ndims...), 
        cov_pars;
        nsamp = nsamp,
        window_buffer = window_buffer,
        target_times = [target_time], 
        spatial_mod = exp_cor,                                           
        obs_operator = unif_weighted_obs_operator_centroid,
        state_in_cov = false,
        cov_wt = 0.2,
        nb_coarse = 2.0);
else
    ## fill in prior mean with mean prior
    nkp = isnan.(prior_mean)
    if sum(nkp) > 0
        mp = nanmean(prior_mean)
        prior_mean[nkp] .= mp
    end

    fused_images, fused_sd_images = coarse_fine_scene_fusion_pmap(fine_data,
        coarse_data,
        fine_geodata, 
        coarse_geodata,
        window_geodata,
        prior_mean,
        prior_sd,
        cov_pars;
        nsamp = nsamp,
        window_buffer = window_buffer,
        target_times = [target_time], 
        spatial_mod = exp_cor,                                           
        obs_operator = unif_weighted_obs_operator_centroid,
        state_in_cov = false,
        cov_wt = 0.2,
        nb_coarse = 2.0);
end;

## remove workers
rmprocs(workers())

if occursin("NDVI", posterior_filename)
    clamp!(fused_images, -1, 1) # NDVI clipped to [-1,1] range
else 
    clamp!(fused_images, 0, 1) # albedo clipped to [0,1]
end

dd = fused_images[:,:,:]

### mask pixels with no historical data 
dd[prior_flag .== 2.0,:] .= NaN # (0,1,2) or (0,1,nan)? nan would be no data
replace!(prior_flag, 2.0 => NaN)

fused_raster = Raster(dd, dims=(x_fine, y_fine, Band(1:1)), missingval=NaN)
flag_raster = Raster(prior_flag, dims=(x_fine, y_fine), missingval=NaN)

@info "writing fused mean: $(posterior_filename)"
write(posterior_filename, fused_raster, force=true)
@info "writing fused mean: $(posterior_flag_filename)"
write(posterior_flag_filename, flag_raster, force=true)
@info "writing fused SD: $(posterior_UQ_filename)"
write(posterior_UQ_filename, Raster(fused_sd_images, dims=(x_fine, y_fine, Band(1:1)), missingval=NaN), force=true)

###save out bias in json
@info "writing bias: $(posterior_bias_filename)"
write_json(posterior_bias_filename, Dict("coarse_bias" => coarse_bias))