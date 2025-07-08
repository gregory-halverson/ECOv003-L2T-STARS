# Define package and environment variables
PACKAGE_NAME = ECOv003-L2T-STARS
ENVIRONMENT_NAME = $(PACKAGE_NAME)
DOCKER_IMAGE_NAME = $(shell echo $(PACKAGE_NAME) | tr '[:upper:]' '[:lower:]')

.PHONY: clean test build twine-upload dist install-package install uninstall reinstall environment remove-environment install-julia colima-start docker-build docker-build-environment docker-build-installation docker-interactive docker-remove

# --- Cleaning and Maintenance ---

clean:
	@echo "Cleaning up build artifacts..."
	rm -rf *.o *.out *.log
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

# --- Testing ---

test:
	@echo "Running tests..."
	PYTHONPATH=. pytest

# --- Package Management ---

build:
	@echo "Building package..."
	python -m build

twine-upload:
	@echo "Uploading package to PyPI..."
	twine upload dist/*

dist: clean build twine-upload
	@echo "Distribution process complete."

install-package:
	@echo "Installing package in development mode..."
	pip install -e .[dev]

install: install-julia install-package
	@echo "All installations complete."

uninstall:
	@echo "Uninstalling $(PACKAGE_NAME)..."
	pip uninstall $(PACKAGE_NAME)

reinstall: uninstall install
	@echo "Package reinstalled."

# --- Environment Management ---

environment:
	@echo "Creating mamba environment: $(ENVIRONMENT_NAME)..."
	mamba create -y -n $(ENVIRONMENT_NAME) -c conda-forge python=3.11

remove-environment:
	@echo "Removing mamba environment: $(ENVIRONMENT_NAME)..."
	mamba env remove -y -n $(ENVIRONMENT_NAME)

install-julia:
	@echo "Installing Julia packages..."
	julia -e 'using Pkg; Pkg.add.(["Glob", "DimensionalData", "HTTP", "JSON", "ArchGDAL", "Rasters", "STARSDataFusion"]); Pkg.develop(path="ECOv003_L2T_STARS/VNP43NRT_jl")'

# --- Docker & Colima ---

colima-start:
	@echo "Starting Colima..."
	colima start -m 16 -a x86_64 -d 100

docker-build:
	@echo "Building Docker image: $(DOCKER_IMAGE_NAME):latest..."
	docker build -t $(DOCKER_IMAGE_NAME):latest .

docker-build-environment:
	@echo "Building Docker environment target..."
	docker build --target environment -t $(DOCKER_IMAGE_NAME):latest .

docker-build-installation:
	@echo "Building Docker installation target..."
	docker build --target installation -t $(DOCKER_IMAGE_NAME):latest .

docker-interactive:
	@echo "Starting interactive Docker session..."
	docker run -it $(DOCKER_IMAGE_NAME) fish

docker-remove:
	@echo "Removing Docker image: $(DOCKER_IMAGE_NAME)..."
	docker rmi -f $(DOCKER_IMAGE_NAME)