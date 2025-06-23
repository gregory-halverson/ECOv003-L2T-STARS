import argparse
import logging
import shutil
import socket
import subprocess
import sys
import urllib
from datetime import datetime, timedelta, date, timezone
from glob import glob
from os import makedirs, remove
from os.path import join, abspath, dirname, expanduser, exists, basename
from shutil import which
from typing import Union
from uuid import uuid4

import colored_logging as cl
import numpy as np
import pandas as pd
import rasters as rt
from dateutil import parser
from dateutil.rrule import rrule, DAILY
from rasters import Raster, RasterGeometry
from scipy import stats

# Custom modules for Harmonized Landsat Sentinel (HLS) and ECOSTRESS data
from harmonized_landsat_sentinel import CMRServerUnreachable
from harmonized_landsat_sentinel import (
    HLSLandsatMissing,
    HLSSentinelMissing,
    HLS,
)
from harmonized_landsat_sentinel import (
    HLSTileNotAvailable,
    HLSSentinelNotAvailable,
    HLSLandsatNotAvailable,
    HLSDownloadFailed,
    HLSNotAvailable,
)
from harmonized_landsat_sentinel import HLSBandNotAcquired, HLS2CMR, CMR_SEARCH_URL

# Custom ECOSTRESS granule and exit code definitions
from ECOv002_granules import L2TLSTE, L2TSTARS, NDVI_COLORMAP, ALBEDO_COLORMAP
from ECOv003_exit_codes import (
    SUCCESS_EXIT_CODE,
    ANCILLARY_SERVER_UNREACHABLE,
    DOWNLOAD_FAILED,
    LAND_FILTER,
    ANCILLARY_LATENCY,
    UNCLASSIFIED_FAILURE_EXIT_CODE,
    RUNCONFIG_FILENAME_NOT_SUPPLIED,
    InputFilesInaccessible,
    MissingRunConfigValue,
    UnableToParseRunConfig,
    BlankOutput,
)

# Custom LPDAAC (Land Processes Distributed Active Archive Center) data pool
from .LPDAAC.LPDAACDataPool import LPDAACServerUnreachable

# Custom VIIRS (Visible Infrared Imaging Radiometer Suite) modules
from .VIIRS import VIIRSDownloaderAlbedo, VIIRSDownloaderNDVI
from .VIIRS.VNP43IA4 import VNP43IA4
from .VIIRS.VNP43MA3 import VNP43MA3
from .VNP43NRT import VNP43NRT

# Custom utility modules
from .daterange import get_date
from .runconfig import ECOSTRESSRunConfig
from .timer import Timer

# --- Global Variables and Constants ---

# Read the version from a version.txt file located in the same directory as this script.
with open(join(abspath(dirname(__file__)), "version.txt")) as f:
    version = f.read().strip()  # .strip() to remove any leading/trailing whitespace

__version__ = version
PGEVersion = __version__  # PGE (Product Generation Executive) Version

# Template file for generating the L2T_STARS run configuration XML.
L2T_STARS_TEMPLATE = join(abspath(dirname(__file__)), "L2T_STARS.xml")

# Default directories and parameters for the L2T_STARS processing.
DEFAULT_WORKING_DIRECTORY = "."  # Current directory
DEFAULT_BUILD = "0700"  # Default build ID
PRIMARY_VARIABLE = "NDVI"  # Primary variable of interest
DEFAULT_OUTPUT_DIRECTORY = "L2T_STARS_output"
DEFAULT_STARS_SOURCES_DIRECTORY = "L2T_STARS_SOURCES"
DEFAULT_STARS_INDICES_DIRECTORY = "L2T_STARS_INDICES"
DEFAULT_STARS_MODEL_DIRECTORY = "L2T_STARS_MODEL"
DEFAULT_STARS_PRODUCTS_DIRECTORY = "STARS_products"
DEFAULT_HLS_DOWNLOAD_DIRECTORY = "HLS2_download"
DEFAULT_LANDSAT_DOWNLOAD_DIRECTORY = "HLS2_download"  # Redundant but kept for clarity
DEFAULT_HLS_PRODUCTS_DIRECTORY = "HLS2_products"
DEFAULT_VIIRS_DOWNLOAD_DIRECTORY = "VIIRS_download"
DEFAULT_VIIRS_PRODUCTS_DIRECTORY = "VIIRS_products"
DEFAUL_VIIRS_MOSAIC_DIRECTORY = "VIIRS_mosaic"
DEFAULT_GEOS5FP_DOWNLOAD_DIRECTORY = "GEOS5FP_download"
DEFAULT_GEOS5FP_PRODUCTS_DIRECTORY = "GEOS5FP_products"
DEFAULT_VNP09GA_PRODUCTS_DIRECTORY = "VNP09GA_products"
DEFAULT_VNP43NRT_PRODUCTS_DIRECTORY = "VNP43NRT_products"

# Processing parameters
VIIRS_GIVEUP_DAYS = 4  # Number of days to give up waiting for VIIRS data
DEFAULT_SPINUP_DAYS = 7  # Spin-up period for time-series analysis
DEFAULT_TARGET_RESOLUTION = 70  # Target output resolution in meters
DEFAULT_NDVI_RESOLUTION = 490  # NDVI coarse resolution in meters
DEFAULT_ALBEDO_RESOLUTION = 980  # Albedo coarse resolution in meters
DEFAULT_USE_SPATIAL = False  # Flag for using spatial interpolation (currently unused)
DEFAULT_USE_VNP43NRT = True  # Flag for using VNP43NRT VIIRS product
DEFAULT_CALIBRATE_FINE = False  # Flag for calibrating fine resolution data to coarse

# Product short and long names
L2T_STARS_SHORT_NAME = "ECO_L2T_STARS"
L2T_STARS_LONG_NAME = "ECOSTRESS Tiled Ancillary NDVI and Albedo L2 Global 70 m"

# Initialize the logger for the module
logger = logging.getLogger(__name__)


# --- Data Structures ---


class Prior:
    """
    A data class to encapsulate information about a prior L2T_STARS product.

    This class holds filenames and flags related to the use of a previous
    STARS product as a 'prior' in the data fusion process. This can help
    to constrain the solution and improve accuracy, especially when
    observations for the current date are sparse.

    Attributes:
        using_prior (bool): True if a prior product is being used, False otherwise.
        prior_date_UTC (date): The UTC date of the prior product.
        L2T_STARS_prior_filename (str): Path to the prior L2T_STARS zip file.
        prior_NDVI_filename (str): Path to the prior NDVI mean file.
        prior_NDVI_UQ_filename (str): Path to the prior NDVI uncertainty (UQ) file.
        prior_NDVI_flag_filename (str): Path to the prior NDVI flag file.
        prior_NDVI_bias_filename (str): Path to the prior NDVI bias file.
        prior_NDVI_bias_UQ_filename (str): Path to the prior NDVI bias uncertainty file.
        prior_albedo_filename (str): Path to the prior albedo mean file.
        prior_albedo_UQ_filename (str): Path to the prior albedo uncertainty (UQ) file.
        prior_albedo_flag_filename (str): Path to the prior albedo flag file.
        prior_albedo_bias_filename (str): Path to the prior albedo bias file.
        prior_albedo_bias_UQ_filename (str): Path to the prior albedo bias uncertainty file.
    """

    def __init__(
        self,
        using_prior: bool = False,
        prior_date_UTC: date = None,
        L2T_STARS_prior_filename: str = None,
        prior_NDVI_filename: str = None,
        prior_NDVI_UQ_filename: str = None,
        prior_NDVI_flag_filename: str = None,
        prior_NDVI_bias_filename: str = None,
        prior_NDVI_bias_UQ_filename: str = None,
        prior_albedo_filename: str = None,
        prior_albedo_UQ_filename: str = None,
        prior_albedo_flag_filename: str = None,
        prior_albedo_bias_filename: str = None,
        prior_albedo_bias_UQ_filename: str = None,
    ):
        self.using_prior = using_prior
        self.prior_date_UTC = prior_date_UTC
        self.L2T_STARS_prior_filename = L2T_STARS_prior_filename
        self.prior_NDVI_filename = prior_NDVI_filename
        self.prior_NDVI_UQ_filename = prior_NDVI_UQ_filename
        self.prior_NDVI_flag_filename = prior_NDVI_flag_filename
        self.prior_NDVI_bias_filename = prior_NDVI_bias_filename
        self.prior_NDVI_bias_UQ_filename = prior_NDVI_bias_UQ_filename
        self.prior_albedo_filename = prior_albedo_filename
        self.prior_albedo_UQ_filename = prior_albedo_UQ_filename
        self.prior_albedo_flag_filename = prior_albedo_flag_filename
        self.prior_albedo_bias_filename = prior_albedo_bias_filename
        self.prior_albedo_bias_UQ_filename = prior_albedo_bias_UQ_filename


# --- Run-Config Management ---


def generate_L2T_STARS_runconfig(
    L2T_LSTE_filename: str,
    prior_L2T_STARS_filename: str = "",
    orbit: int = None,
    scene: int = None,
    tile: str = None,
    time_UTC: Union[datetime, str] = None,
    working_directory: str = None,
    sources_directory: str = None,
    indices_directory: str = None,
    model_directory: str = None,
    executable_filename: str = None,
    output_directory: str = None,
    runconfig_filename: str = None,
    log_filename: str = None,
    build: str = None,
    processing_node: str = None,
    production_datetime: datetime = None,
    job_ID: str = None,
    instance_ID: str = None,
    product_counter: int = None,
    template_filename: str = None,
) -> str:
    """
    Generates an XML run-configuration file for the L2T_STARS processing.

    This function dynamically creates an XML run-config file by populating a template
    with provided or default parameters. It also checks for and returns existing
    run-configs to prevent redundant generation.

    Args:
        L2T_LSTE_filename (str): Path to the input ECOSTRESS L2T LSTE granule file.
        prior_L2T_STARS_filename (str, optional): Path to a prior L2T_STARS product file.
            Defaults to "".
        orbit (int, optional): Orbit number. If None, derived from L2T_LSTE_filename.
        scene (int, optional): Scene ID. If None, derived from L2T_LSTE_filename.
        tile (str, optional): HLS tile ID (e.g., 'H09V05'). If None, derived from L2T_LSTE_filename.
        time_UTC (Union[datetime, str], optional): UTC time of the L2T_LSTE granule. If None,
            derived from L2T_LSTE_filename.
        working_directory (str, optional): Root directory for all processing outputs.
            Defaults to ".".
        sources_directory (str, optional): Directory for downloaded source data (HLS, VIIRS).
        indices_directory (str, optional): Directory for intermediate index products.
        model_directory (str, optional): Directory for model state files (priors, posteriors).
        executable_filename (str, optional): Path to the L2T_STARS executable. If None,
            'L2T_STARS' is assumed to be in the system's PATH.
        output_directory (str, optional): Directory for final L2T_STARS products.
        runconfig_filename (str, optional): Specific filename for the generated run-config.
            If None, a default name based on granule ID is used.
        log_filename (str, optional): Specific filename for the log file. If None,
            a default name based on granule ID is used.
        build (str, optional): Build ID of the PGE. Defaults to DEFAULT_BUILD.
        processing_node (str, optional): Name of the processing node. Defaults to system hostname.
        production_datetime (datetime, optional): Production date and time. Defaults to now (UTC).
        job_ID (str, optional): Job identifier. Defaults to a timestamp.
        instance_ID (str, optional): Unique instance identifier. Defaults to a UUID.
        product_counter (int, optional): Counter for product generation. Defaults to 1.
        template_filename (str, optional): Path to the XML run-config template file.
            Defaults to L2T_STARS_TEMPLATE.

    Returns:
        str: The absolute path to the generated or existing run-configuration file.
    """
    timer = Timer()  # Start a timer to track function execution time

    # Load the L2T_LSTE granule to extract necessary metadata if not provided
    l2t_lste_granule = L2TLSTE(L2T_LSTE_filename)

    # Use values from L2T_LSTE granule if not explicitly provided
    if orbit is None:
        orbit = l2t_lste_granule.orbit
    if scene is None:
        scene = l2t_lste_granule.scene
    if tile is None:
        tile = l2t_lste_granule.tile
    if time_UTC is None:
        time_UTC = l2t_lste_granule.time_UTC

    # Set default values for other parameters if not provided
    if build is None:
        build = DEFAULT_BUILD
    if working_directory is None:
        working_directory = "."

    date_UTC = time_UTC.date()

    logger.info(
        f"Started generating L2T_STARS run-config for tile {cl.val(tile)} on date {cl.time(date_UTC)}"
    )

    # Check for previous run-configs to avoid re-generating
    pattern = join(
        working_directory, "runconfig", f"ECOv003_L2T_STARS_{tile}_*_{build}_*.xml"
    )
    logger.info(f"Scanning for previous run-configs: {cl.val(pattern)}")
    previous_runconfigs = glob(pattern)
    previous_runconfig_count = len(previous_runconfigs)

    if previous_runconfig_count > 0:
        logger.info(f"Found {cl.val(previous_runconfig_count)} previous run-configs")
        # Return the most recent run-config if found
        previous_runconfig = sorted(previous_runconfigs)[-1]
        logger.info(f"Previous run-config: {cl.file(previous_runconfig)}")
        return previous_runconfig

    # Resolve the path to the run-config template
    if template_filename is None:
        template_filename = L2T_STARS_TEMPLATE
    template_filename = abspath(expanduser(template_filename))

    # Set production datetime if not provided
    if production_datetime is None:
        production_datetime = datetime.now(timezone.utc)

    # Set product counter if not provided
    if product_counter is None:
        product_counter = 1

    # Format timestamp and generate granule ID
    timestamp = f"{time_UTC:%Y%m%d}"
    granule_ID = (
        f"ECOv003_L2T_STARS_{tile}_{timestamp}_{build}_{product_counter:02d}"
    )

    # Define run-config filename and resolve absolute path
    if runconfig_filename is None:
        runconfig_filename = join(working_directory, "runconfig", f"{granule_ID}.xml")
    runconfig_filename = abspath(expanduser(runconfig_filename))

    # If the run-config file already exists, log and return its path
    if exists(runconfig_filename):
        logger.info(f"Run-config already exists {cl.file(runconfig_filename)}")
        return runconfig_filename

    # Resolve absolute paths for various directories if not already defined
    working_directory = abspath(expanduser(working_directory))
    if sources_directory is None:
        sources_directory = join(working_directory, DEFAULT_STARS_SOURCES_DIRECTORY)
    if indices_directory is None:
        indices_directory = join(working_directory, DEFAULT_STARS_INDICES_DIRECTORY)
    if model_directory is None:
        model_directory = join(working_directory, DEFAULT_STARS_MODEL_DIRECTORY)

    # Determine executable path; fall back to just the name if not found in PATH
    if executable_filename is None:
        executable_filename = which("L2T_STARS")
    if executable_filename is None:
        executable_filename = "L2T_STARS"

    # Define output and log file paths
    if output_directory is None:
        output_directory = join(working_directory, DEFAULT_OUTPUT_DIRECTORY)
    output_directory = abspath(expanduser(output_directory))
    if log_filename is None:
        log_filename = join(working_directory, "log", f"{granule_ID}.log")
    log_filename = abspath(expanduser(log_filename))

    # Get processing node hostname
    if processing_node is None:
        processing_node = socket.gethostname()

    # Set Job ID and Instance ID
    if job_ID is None:
        job_ID = timestamp
    if instance_ID is None:
        instance_ID = str(uuid4())  # Generate a unique UUID for the instance

    # Resolve absolute path for the input L2T_LSTE file
    L2T_LSTE_filename = abspath(expanduser(L2T_LSTE_filename))

    logger.info(f"Loading L2T_STARS template: {cl.file(template_filename)}")

    # Read the XML template file content
    with open(template_filename, "r") as file:
        template = file.read()

    # Replace placeholders in the template with actual values
    logger.info(f"Orbit: {cl.val(orbit)}")
    template = template.replace("orbit_number", f"{orbit:05d}")
    logger.info(f"Scene: {cl.val(scene)}")
    template = template.replace("scene_ID", f"{scene:03d}")
    logger.info(f"Tile: {cl.val(tile)}")
    template = template.replace("tile_ID", f"{tile}")
    logger.info(f"L2T_LSTE file: {cl.file(L2T_LSTE_filename)}")
    template = template.replace("L2T_LSTE_filename", L2T_LSTE_filename)
    logger.info(f"Prior L2T_STARS file: {cl.file(prior_L2T_STARS_filename)}")
    template = template.replace("prior_L2T_STARS_filename", prior_L2T_STARS_filename)
    logger.info(f"Working directory: {cl.dir(working_directory)}")
    template = template.replace("working_directory", working_directory)
    logger.info(f"Sources directory: {cl.dir(sources_directory)}")
    template = template.replace("sources_directory", sources_directory)
    logger.info(f"Indices directory: {cl.dir(indices_directory)}")
    template = template.replace("indices_directory", indices_directory)
    logger.info(f"Model directory: {cl.dir(model_directory)}")
    template = template.replace("model_directory", model_directory)
    logger.info(f"Executable: {cl.file(executable_filename)}")
    template = template.replace("executable_filename", executable_filename)
    logger.info(f"Output directory: {cl.dir(output_directory)}")
    template = template.replace("output_directory", output_directory)
    logger.info(f"Run-config: {cl.file(runconfig_filename)}")
    template = template.replace("runconfig_filename", runconfig_filename)
    logger.info(f"Log: {cl.file(log_filename)}")
    template = template.replace("log_filename", log_filename)
    logger.info(f"Build: {cl.val(build)}")
    template = template.replace("build_ID", build)
    logger.info(f"Processing node: {cl.val(processing_node)}")
    template = template.replace("processing_node", processing_node)
    logger.info(f"Production date/time: {cl.time(production_datetime)}")
    template = template.replace("production_datetime", f"{production_datetime:%Y-%m-%dT%H:%M:%SZ}")
    logger.info(f"Job ID: {cl.val(job_ID)}")
    template = template.replace("job_ID", job_ID)
    logger.info(f"Instance ID: {cl.val(instance_ID)}")
    template = template.replace("instance_ID", instance_ID)
    logger.info(f"Product counter: {cl.val(product_counter)}")
    template = template.replace("product_counter", f"{product_counter:02d}")

    # Create the directory for the run-config file if it doesn't exist
    makedirs(dirname(abspath(runconfig_filename)), exist_ok=True)
    logger.info(f"Writing run-config file: {cl.file(runconfig_filename)}")

    # Write the populated template to the run-config file
    with open(runconfig_filename, "w") as file:
        file.write(template)

    logger.info(
        f"Finished generating L2T_STARS run-config for orbit {cl.val(orbit)} scene {cl.val(scene)} ({timer})"
    )

    return runconfig_filename


class L2TSTARSConfig(ECOSTRESSRunConfig):
    """
    Parses and validates the L2T_STARS specific parameters from an XML run-configuration file.

    This class extends the base ECOSTRESSRunConfig to extract paths, IDs,
    and processing parameters relevant to the L2T_STARS product generation.
    It performs validation to ensure all critical parameters are present.
    """

    def __init__(self, filename: str):
        """
        Initializes the L2TSTARSConfig by parsing the provided run-config XML file.

        Args:
            filename (str): The path to the L2T_STARS run-configuration XML file.

        Raises:
            MissingRunConfigValue: If a required value is missing from the run-config.
            UnableToParseRunConfig: If the run-config file cannot be parsed due to other errors.
        """
        logger.info(f"Loading L2T_STARS run-config: {cl.file(filename)}")
        # Read the run-config XML into a dictionary
        runconfig = self.read_runconfig(filename)

        try:
            # Validate and extract working directory from StaticAncillaryFileGroup
            if "StaticAncillaryFileGroup" not in runconfig:
                raise MissingRunConfigValue(
                    f"Missing StaticAncillaryFileGroup in L2T_STARS run-config: {filename}"
                )
            if "L2T_STARS_WORKING" not in runconfig["StaticAncillaryFileGroup"]:
                raise MissingRunConfigValue(
                    f"Missing StaticAncillaryFileGroup/L2T_STARS_WORKING in L2T_STARS run-config: {filename}"
                )
            self.working_directory = abspath(
                runconfig["StaticAncillaryFileGroup"]["L2T_STARS_WORKING"]
            )
            logger.info(f"Working directory: {cl.dir(self.working_directory)}")

            # Validate and extract sources directory
            if "L2T_STARS_SOURCES" not in runconfig["StaticAncillaryFileGroup"]:
                raise MissingRunConfigValue(
                    f"Missing StaticAncillaryFileGroup/L2T_STARS_SOURCES in L2T_STARS run-config: {filename}"
                )
            self.sources_directory = abspath(
                runconfig["StaticAncillaryFileGroup"]["L2T_STARS_SOURCES"]
            )
            logger.info(f"Sources directory: {cl.dir(self.sources_directory)}")

            # Validate and extract indices directory
            if "L2T_STARS_INDICES" not in runconfig["StaticAncillaryFileGroup"]:
                raise MissingRunConfigValue(
                    f"Missing StaticAncillaryFileGroup/L2T_STARS_INDICES in L2T_STARS run-config: {filename}"
                )
            self.indices_directory = abspath(
                runconfig["StaticAncillaryFileGroup"]["L2T_STARS_INDICES"]
            )
            logger.info(f"Indices directory: {cl.dir(self.indices_directory)}")

            # Validate and extract model directory
            if "L2T_STARS_MODEL" not in runconfig["StaticAncillaryFileGroup"]:
                raise MissingRunConfigValue(
                    f"Missing StaticAncillaryFileGroup/L2T_STARS_MODEL in L2T_STARS run-config: {filename}"
                )
            self.model_directory = abspath(
                runconfig["StaticAncillaryFileGroup"]["L2T_STARS_MODEL"]
            )
            logger.info(f"Model directory: {cl.dir(self.model_directory)}")

            # Validate and extract output directory from ProductPathGroup
            if "ProductPathGroup" not in runconfig:
                raise MissingRunConfigValue(
                    f"Missing ProductPathGroup in L2T_STARS run-config: {filename}"
                )
            if "ProductPath" not in runconfig["ProductPathGroup"]:
                raise MissingRunConfigValue(
                    f"Missing ProductPathGroup/ProductPath in L2T_STARS run-config: {filename}"
                )
            self.output_directory = abspath(
                runconfig["ProductPathGroup"]["ProductPath"]
            )
            logger.info(f"Output directory: {cl.dir(self.output_directory)}")

            # Validate and extract input L2T_LSTE filename
            if "InputFileGroup" not in runconfig:
                raise MissingRunConfigValue(
                    f"Missing InputFileGroup in L2G_L2T_LSTE run-config: {filename}"
                )
            if "L2T_LSTE" not in runconfig["InputFileGroup"]:
                raise MissingRunConfigValue(
                    f"Missing InputFileGroup/L2T_LSTE in L2T_STARS run-config: {filename}"
                )
            self.L2T_LSTE_filename = abspath(runconfig["InputFileGroup"]["L2T_LSTE"])
            logger.info(f"L2T_LSTE file: {cl.file(self.L2T_LSTE_filename)}")

            # Extract optional prior L2T_STARS filename
            self.L2T_STARS_prior_filename = None
            if "L2T_STARS_PRIOR" in runconfig["InputFileGroup"]:
                prior_filename = runconfig["InputFileGroup"]["L2T_STARS_PRIOR"]
                if prior_filename != "" and exists(prior_filename):
                    self.L2T_STARS_prior_filename = abspath(prior_filename)
                logger.info(
                    f"L2T_STARS prior file: {cl.file(self.L2T_STARS_prior_filename)}"
                )

            # Extract geometry parameters (orbit, scene, tile)
            self.orbit = int(runconfig["Geometry"]["OrbitNumber"])
            logger.info(f"Orbit: {cl.val(self.orbit)}")
            if "SceneId" not in runconfig["Geometry"]:
                raise MissingRunConfigValue(
                    f"Missing Geometry/SceneId in L2T_STARS run-config: {filename}"
                )
            self.scene = int(runconfig["Geometry"]["SceneId"])
            logger.info(f"Scene: {cl.val(self.scene)}")
            if "TileId" not in runconfig["Geometry"]:
                raise MissingRunConfigValue(
                    f"Missing Geometry/TileId in L2T_STARS run-config: {filename}"
                )
            self.tile = str(runconfig["Geometry"]["TileId"])
            logger.info(f"Tile: {cl.val(self.tile)}")

            # Extract production details
            if "ProductionDateTime" not in runconfig["JobIdentification"]:
                raise MissingRunConfigValue(
                    f"Missing JobIdentification/ProductionDateTime in L2T_STARS run-config {filename}"
                )
            self.production_datetime = parser.parse(
                runconfig["JobIdentification"]["ProductionDateTime"]
            )
            logger.info(f"Production time: {cl.time(self.production_datetime)}")

            # Extract build ID
            if "BuildID" not in runconfig["PrimaryExecutable"]:
                raise MissingRunConfigValue(
                    f"Missing PrimaryExecutable/BuildID in L2T_STARS run-config {filename}"
                )
            self.build = str(runconfig["PrimaryExecutable"]["BuildID"])

            # Extract product counter
            if "ProductCounter" not in runconfig["ProductPathGroup"]:
                raise MissingRunConfigValue(
                    f"Missing ProductPathGroup/ProductCounter in L2T_STARS run-config {filename}"
                )
            self.product_counter = int(runconfig["ProductPathGroup"]["ProductCounter"])

            # Get UTC time from the L2T_LSTE granule itself
            l2t_lste_granule_obj = L2TLSTE(self.L2T_LSTE_filename)
            time_UTC = l2t_lste_granule_obj.time_UTC

            # Construct the full granule ID and paths for the output product
            granule_ID = (
                f"ECOv003_L2T_STARS_{self.tile}_{time_UTC:%Y%m%d}_{self.build}_"
                f"{self.product_counter:02d}"
            )
            self.granule_ID = granule_ID
            self.L2T_STARS_granule_directory = join(self.output_directory, granule_ID)
            self.L2T_STARS_zip_filename = f"{self.L2T_STARS_granule_directory}.zip"
            self.L2T_STARS_browse_filename = (
                f"{self.L2T_STARS_granule_directory}.png"
            )

        except MissingRunConfigValue as e:
            # Re-raise specific missing value errors
            raise e
        except ECOSTRESSExitCodeException as e:
            # Re-raise custom ECOSTRESS exit code exceptions
            raise e
        except Exception as e:
            # Catch any other parsing errors and raise a generic UnableToParseRunConfig
            logger.exception(e)
            raise UnableToParseRunConfig(
                f"Unable to parse run-config file: {filename}"
            )


# --- File and Directory Management Utilities ---


def generate_filename(
    directory: str, variable: str, date_UTC: Union[date, str], tile: str, cell_size: int
) -> str:
    """
    Generates a standardized filename for a raster product and ensures its directory exists.

    The filename format is STARS_{variable}_{YYYY-MM-DD}_{tile}_{cell_size}m.tif.

    Args:
        directory (str): The base directory where the file will be saved.
        variable (str): The name of the variable (e.g., "NDVI", "albedo").
        date_UTC (Union[date, str]): The UTC date of the data. Can be a date object or a string.
        tile (str): The HLS tile ID (e.g., 'H09V05').
        cell_size (int): The spatial resolution in meters (e.g., 70, 490, 980).

    Returns:
        str: The full, standardized path to the generated filename.
    """
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    variable = str(variable)
    timestamp = date_UTC.strftime("%Y-%m-%d")
    tile = str(tile)
    cell_size = int(cell_size)
    filename = join(directory, f"STARS_{variable}_{timestamp}_{tile}_{cell_size}m.tif")
    # Ensure the directory structure for the file exists
    makedirs(dirname(filename), exist_ok=True)

    return filename


def calibrate_fine_to_coarse(fine_image: Raster, coarse_image: Raster) -> Raster:
    """
    Calibrates a fine-resolution raster image to a coarse-resolution raster image
    using linear regression.

    This function aggregates the fine image to the geometry of the coarse image,
    then performs a linear regression between the aggregated fine image and the
    original coarse image. The derived slope and intercept are then applied to
    the original fine image for calibration.

    Args:
        fine_image (Raster): The higher-resolution raster image to be calibrated.
        coarse_image (Raster): The lower-resolution raster image used as the reference
                                for calibration.

    Returns:
        Raster: The calibrated fine-resolution raster image. If too few valid
                data points are available for regression (less than 30), the
                original fine_image is returned.
    """
    # Aggregate the fine image to the coarse image's geometry for comparison
    aggregated_image = fine_image.to_geometry(coarse_image.geometry, resampling="average")
    x = np.array(aggregated_image).flatten()  # Independent variable (aggregated fine)
    y = np.array(coarse_image).flatten()  # Dependent variable (coarse)

    # Create a mask to remove NaN values from both arrays, ensuring valid data points for regression
    mask = ~np.isnan(x) & ~np.isnan(y)

    # Check if there are enough valid data points for a meaningful linear regression
    if np.count_nonzero(mask) < 30:
        logger.warning(
            f"Insufficient valid data points ({np.count_nonzero(mask)}) for calibration. "
            "Returning original fine image."
        )
        return fine_image

    # Apply the mask to get only valid data points
    x = x[mask]
    y = y[mask]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    logger.info(
        f"Linear regression for calibration: slope={slope:.4f}, intercept={intercept:.4f}, "
        f"R-squared={r_value**2:.4f}"
    )

    # Apply the derived calibration to the original fine image
    calibrated_image = fine_image * slope + intercept

    return calibrated_image


def generate_NDVI_coarse_image(
    date_UTC: Union[date, str], VIIRS_connection: VIIRSDownloaderNDVI, geometry: RasterGeometry = None
) -> Raster:
    """
    Generates a coarse-resolution NDVI image from VIIRS data.

    Args:
        date_UTC (Union[date, str]): The UTC date for which to retrieve NDVI data.
        VIIRS_connection (VIIRSDownloaderNDVI): An initialized VIIRS NDVI downloader object.
        geometry (RasterGeometry, optional): The target geometry for the VIIRS image.
                                            If None, the native VIIRS geometry is used.

    Returns:
        Raster: A Raster object representing the coarse-resolution NDVI image.
                Zero values are converted to NaN.
    """
    coarse_image = VIIRS_connection.NDVI(date_UTC=date_UTC, geometry=geometry)
    # Convert zero values (often used as NoData in some datasets) to NaN for proper handling
    coarse_image = rt.where(coarse_image == 0, np.nan, coarse_image)
    return coarse_image


def generate_NDVI_fine_image(
    date_UTC: Union[date, str], tile: str, HLS_connection: HLS
) -> Raster:
    """
    Generates a fine-resolution NDVI image from HLS data.

    Args:
        date_UTC (Union[date, str]): The UTC date for which to retrieve NDVI data.
        tile (str): The HLS tile ID.
        HLS_connection (HLS): An initialized HLS data connection object.

    Returns:
        Raster: A Raster object representing the fine-resolution NDVI image.
                Zero values are converted to NaN.
    """
    fine_image = HLS_connection.NDVI(tile=tile, date_UTC=date_UTC)
    # Convert zero values to NaN for consistency
    fine_image = rt.where(fine_image == 0, np.nan, fine_image)
    return fine_image


def generate_albedo_coarse_image(
    date_UTC: Union[date, str], VIIRS_connection: VIIRSDownloaderAlbedo, geometry: RasterGeometry = None
) -> Raster:
    """
    Generates a coarse-resolution albedo image from VIIRS data.

    Args:
        date_UTC (Union[date, str]): The UTC date for which to retrieve albedo data.
        VIIRS_connection (VIIRSDownloaderAlbedo): An initialized VIIRS albedo downloader object.
        geometry (RasterGeometry, optional): The target geometry for the VIIRS image.
                                            If None, the native VIIRS geometry is used.

    Returns:
        Raster: A Raster object representing the coarse-resolution albedo image.
                Zero values are converted to NaN.
    """
    coarse_image = VIIRS_connection.albedo(date_UTC=date_UTC, geometry=geometry)
    # Convert zero values to NaN for consistency
    coarse_image = rt.where(coarse_image == 0, np.nan, coarse_image)
    return coarse_image


def generate_albedo_fine_image(
    date_UTC: Union[date, str], tile: str, HLS_connection: HLS
) -> Raster:
    """
    Generates a fine-resolution albedo image from HLS data.

    Args:
        date_UTC (Union[date, str]): The UTC date for which to retrieve albedo data.
        tile (str): The HLS tile ID.
        HLS_connection (HLS): An initialized HLS data connection object.

    Returns:
        Raster: A Raster object representing the fine-resolution albedo image.
                Zero values are converted to NaN.
    """
    fine_image = HLS_connection.albedo(tile=tile, date_UTC=date_UTC)
    # Convert zero values to NaN for consistency
    fine_image = rt.where(fine_image == 0, np.nan, fine_image)
    return fine_image


def generate_input_staging_directory(
    input_staging_directory: str, tile: str, prefix: str
) -> str:
    """
    Generates a path for an input staging directory and ensures it exists.

    This is used to organize temporary input files for the Julia processing.

    Args:
        input_staging_directory (str): The base input staging directory.
        tile (str): The HLS tile ID.
        prefix (str): A prefix for the sub-directory name (e.g., "NDVI_coarse").

    Returns:
        str: The full path to the created or existing staging directory.
    """
    directory = join(input_staging_directory, f"{prefix}_{tile}")
    makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
    return directory


def generate_NDVI_coarse_directory(input_staging_directory: str, tile: str) -> str:
    """
    Generates the specific staging directory for coarse NDVI images.

    Args:
        input_staging_directory (str): The base input staging directory.
        tile (str): The HLS tile ID.

    Returns:
        str: The full path to the coarse NDVI staging directory.
    """
    return generate_input_staging_directory(input_staging_directory, tile, "NDVI_coarse")


def generate_NDVI_fine_directory(input_staging_directory: str, tile: str) -> str:
    """
    Generates the specific staging directory for fine NDVI images.

    Args:
        input_staging_directory (str): The base input staging directory.
        tile (str): The HLS tile ID.

    Returns:
        str: The full path to the fine NDVI staging directory.
    """
    return generate_input_staging_directory(input_staging_directory, tile, "NDVI_fine")


def generate_albedo_coarse_directory(input_staging_directory: str, tile: str) -> str:
    """
    Generates the specific staging directory for coarse albedo images.

    Args:
        input_staging_directory (str): The base input staging directory.
        tile (str): The HLS tile ID.

    Returns:
        str: The full path to the coarse albedo staging directory.
    """
    return generate_input_staging_directory(
        input_staging_directory, tile, "albedo_coarse"
    )


def generate_albedo_fine_directory(input_staging_directory: str, tile: str) -> str:
    """
    Generates the specific staging directory for fine albedo images.

    Args:
        input_staging_directory (str): The base input staging directory.
        tile (str): The HLS tile ID.

    Returns:
        str: The full path to the fine albedo staging directory.
    """
    return generate_input_staging_directory(
        input_staging_directory, tile, "albedo_fine"
    )


def generate_output_directory(
    working_directory: str, date_UTC: Union[date, str], tile: str
) -> str:
    """
    Generates a dated output directory for Julia model results and ensures it exists.

    Args:
        working_directory (str): The main working directory.
        date_UTC (Union[date, str]): The UTC date for the output.
        tile (str): The HLS tile ID.

    Returns:
        str: The full path to the created or existing output directory.
    """
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    directory = join(working_directory, f"julia_output_{date_UTC:%y.%m.%d}_{tile}")
    makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
    return directory


def generate_model_state_tile_date_directory(
    model_directory: str, tile: str, date_UTC: Union[date, str]
) -> str:
    """
    Generates a directory for storing model state files (e.g., priors, posteriors)
    organized by tile and date, and ensures it exists.

    Args:
        model_directory (str): The base directory for model state files.
        tile (str): The HLS tile ID.
        date_UTC (Union[date, str]): The UTC date for the model state.

    Returns:
        str: The full path to the created or existing model state directory.
    """
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    directory = join(model_directory, tile, f"{date_UTC:%Y-%m-%d}")
    makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
    return directory


# --- Julia Integration Functions ---


def install_STARS_jl(
    github_URL: str = "https://github.com/STARS-Data-Fusion/STARS.jl",
    environment_name: str = "@ECOv003-L2T-STARS",
) -> subprocess.CompletedProcess:
    """
    Installs the STARS.jl Julia package from GitHub into a specified Julia environment.

    This function executes a Julia command to activate a given environment and
    then develops (installs in editable mode) the STARS.jl package from its
    GitHub repository.

    Args:
        github_URL (str, optional): The URL of the GitHub repository containing STARS.jl.
                                    Defaults to "https://github.com/STARS-Data-Fusion/STARS.jl".
        environment_name (str, optional): The name of the Julia environment to install
                                          the package into. Defaults to "@ECOv003-L2T-STARS".

    Returns:
        subprocess.CompletedProcess: An object containing information about the
                                     execution of the Julia command (return code, stdout, stderr).
    """
    # Julia command to activate an environment and then add/develop a package from URL
    julia_command = [
        "julia",
        "-e",
        f'using Pkg; Pkg.activate("{environment_name}"); Pkg.develop(url="{github_URL}")',
    ]

    # Execute the Julia command as a subprocess
    result = subprocess.run(julia_command, capture_output=True, text=True, check=False)

    if result.returncode == 0:
        logger.info(
            f"STARS.jl installed successfully in environment '{environment_name}'!"
        )
    else:
        logger.error("Error installing STARS.jl:")
        logger.error(result.stderr)
    return result


def instantiate_STARS_jl(package_location: str) -> subprocess.CompletedProcess:
    """
    Activates a Julia project at a given location and instantiates its dependencies.

    This is necessary to ensure all required Julia packages for STARS.jl are
    downloaded and ready for use within the specified project environment.

    Args:
        package_location (str): The directory of the Julia package (where Project.toml is located)
                                to activate and instantiate.

    Returns:
        subprocess.CompletedProcess: An object containing information about the
                                     execution of the Julia command (return code, stdout, stderr).
    """
    # Julia command to activate a specific package location and then instantiate its dependencies
    julia_command = [
        "julia",
        "-e",
        f'using Pkg; Pkg.activate("{package_location}"); Pkg.instantiate()',
    ]

    # Execute the Julia command as a subprocess
    result = subprocess.run(julia_command, capture_output=True, text=True, check=False)

    if result.returncode == 0:
        logger.info(
            f"STARS.jl instantiated successfully in directory '{package_location}'!"
        )
    else:
        logger.error("Error instantiating STARS.jl:")
        logger.error(result.stderr)
    return result


def process_julia_data_fusion(
    tile: str,
    coarse_cell_size: int,
    fine_cell_size: int,
    VIIRS_start_date: date,
    VIIRS_end_date: date,
    HLS_start_date: date,
    HLS_end_date: date,
    coarse_directory: str,
    fine_directory: str,
    posterior_filename: str,
    posterior_UQ_filename: str,
    posterior_flag_filename: str,
    posterior_bias_filename: str,
    posterior_bias_UQ_filename: str,
    prior_filename: str = None,
    prior_UQ_filename: str = None,
    prior_bias_filename: str = None,
    prior_bias_UQ_filename: str = None,
    environment_name: str = "@ECOv003-L2T-STARS",  # Unused in current Julia command, but kept for consistency
    threads: Union[int, str] = "auto",
    num_workers: int = 4,
):
    """
    Executes the Julia-based data fusion process for NDVI or albedo.

    This function prepares and runs a Julia script that performs the core
    STARS data fusion. It passes all necessary input and output paths,
    date ranges, and resolution parameters to the Julia script. Optionally,
    it can also pass prior information to the Julia system.

    Args:
        tile (str): The HLS tile ID.
        coarse_cell_size (int): The cell size of the coarse resolution data (e.g., VIIRS).
        fine_cell_size (int): The cell size of the fine resolution data (e.g., HLS and target).
        VIIRS_start_date (date): Start date for VIIRS data processing.
        VIIRS_end_date (date): End date for VIIRS data processing.
        HLS_start_date (date): Start date for HLS data processing.
        HLS_end_date (date): End date for HLS data processing.
        coarse_directory (str): Directory containing coarse resolution input images.
        fine_directory (str): Directory containing fine resolution input images.
        posterior_filename (str): Output path for the fused posterior mean image.
        posterior_UQ_filename (str): Output path for the fused posterior uncertainty image.
        posterior_flag_filename (str): Output path for the fused posterior flag image.
        posterior_bias_filename (str): Output path for the fused posterior bias image.
        posterior_bias_UQ_filename (str): Output path for the fused posterior bias uncertainty image.
        prior_filename (str, optional): Path to the prior mean image. Defaults to None.
        prior_UQ_filename (str, optional): Path to the prior uncertainty image. Defaults to None.
        prior_bias_filename (str, optional): Path to the prior bias image. Defaults to None.
        prior_bias_UQ_filename (str, optional): Path to the prior bias uncertainty image. Defaults to None.
        environment_name (str, optional): Julia environment name. Defaults to "@ECOv003-L2T-STARS".
        threads (Union[int, str], optional): Number of Julia threads to use, or "auto".
                                            Defaults to "auto".
        num_workers (int, optional): Number of Julia workers for distributed processing.
                                     Defaults to 4.
    """
    # Construct the path to the Julia processing script
    julia_script_filename = join(
        abspath(dirname(__file__)), "process_ECOSTRESS_data_fusion_distributed_bias.jl"
    )
    # The directory where the Julia Project.toml is located
    STARS_source_directory = abspath(dirname(__file__))

    # Instantiate Julia dependencies
    instantiate_STARS_jl(STARS_source_directory)

    # Base Julia command with required arguments
    command = (
        f'export JULIA_NUM_THREADS={threads}; julia --threads {threads} '
        f'"{julia_script_filename}" {num_workers} "{tile}" "{coarse_cell_size}" '
        f'"{fine_cell_size}" "{VIIRS_start_date}" "{VIIRS_end_date}" '
        f'"{HLS_start_date}" "{HLS_end_date}" "{coarse_directory}" '
        f'"{fine_directory}" "{posterior_filename}" "{posterior_UQ_filename}" '
        f'"{posterior_flag_filename}" "{posterior_bias_filename}" '
        f'"{posterior_bias_UQ_filename}"'
    )

    # Conditionally add prior arguments if all prior filenames are provided and exist
    if all(
        [
            filename is not None and exists(filename)
            for filename in [
                prior_filename,
                prior_UQ_filename,
                prior_bias_filename,
                prior_bias_UQ_filename,
            ]
        ]
    ):
        logger.info("Passing prior into Julia data fusion system")
        command += (
            f' "{prior_filename}" "{prior_UQ_filename}" "{prior_bias_filename}" '
            f'"{prior_bias_UQ_filename}"'
        )
    else:
        logger.info("No complete prior set found; running Julia data fusion without prior.")

    logger.info(f"Executing Julia command: {command}")
    # Execute the Julia command. Using shell=True as the command string includes shell syntax (export).
    # This assumes the Julia executable is in the system's PATH.
    subprocess.run(command, shell=True, check=False)


# --- Data Retrieval Functions ---


def retrieve_STARS_sources(
    tile: str,
    geometry: RasterGeometry,
    HLS_start_date: date,
    HLS_end_date: date,
    VIIRS_start_date: date,
    VIIRS_end_date: date,
    HLS_connection: HLS2CMR,
    VIIRS_connection: VNP43NRT,  # Using VNP43NRT as a representative VIIRS connection
):
    """
    Retrieves necessary Harmonized Landsat Sentinel (HLS) and VIIRS source data.

    This function downloads HLS Sentinel and Landsat data, and prefetches VIIRS VNP09GA
    data for the specified tile and date ranges. It includes error handling for
    download failures and data unavailability.

    Args:
        tile (str): The HLS tile ID.
        geometry (RasterGeometry): The spatial geometry of the area of interest.
        HLS_start_date (date): The start date for HLS data retrieval.
        HLS_end_date (date): The end date for HLS data retrieval.
        VIIRS_start_date (date): The start date for VIIRS data retrieval.
        VIIRS_end_date (date): The end date for VIIRS data retrieval.
        HLS_connection (HLS2CMR): An initialized HLS data connection object.
        VIIRS_connection (VNP43NRT): An initialized VIIRS data connection object
                                      (can be VNP43NRT, VNP43IA4, or VNP43MA3).

    Raises:
        DownloadFailed: If an HLS download fails.
        AncillaryLatency: If HLS data for a given tile/date is not available (latency issue).
    """
    logger.info(
        f"Retrieving HLS sources for tile {cl.place(tile)} from {cl.time(HLS_start_date)} to {cl.time(HLS_end_date)}"
    )
    # Iterate through each day in the HLS date range to retrieve Sentinel and Landsat data
    for processing_date in [
        get_date(dt) for dt in rrule(DAILY, dtstart=HLS_start_date, until=HLS_end_date)
    ]:
        try:
            logger.info(
                f"Retrieving HLS Sentinel at tile {cl.place(tile)} on date {cl.time(processing_date)}"
            )
            # Attempt to download HLS Sentinel data
            HLS_connection.sentinel(tile=tile, date_UTC=processing_date)
            logger.info(
                f"Retrieving HLS Landsat at tile {cl.place(tile)} on date {cl.time(processing_date)}"
            )
            # Attempt to download HLS Landsat data
            HLS_connection.landsat(tile=tile, date_UTC=processing_date)
        except HLSDownloadFailed as e:
            logger.exception(e)
            raise DownloadFailed(e)
        except (
            HLSTileNotAvailable,
            HLSSentinelNotAvailable,
            HLSLandsatNotAvailable,
        ) as e:
            # Log warnings for data not being available, but continue processing
            logger.warning(e)
        except Exception as e:
            # Catch other unexpected exceptions during HLS retrieval
            logger.warning("Exception raised while retrieving HLS tiles")
            logger.exception(e)
            continue  # Continue to the next date even if one HLS retrieval fails

    logger.info(
        f"Retrieving VIIRS sources for tile {cl.place(tile)} from {cl.time(VIIRS_start_date)} to {cl.time(VIIRS_end_date)}"
    )
    # Prefetch VNP09GA data for the specified VIIRS date range and geometry
    VIIRS_connection.prefetch_VNP09GA(
        start_date=VIIRS_start_date,
        end_date=VIIRS_end_date,
        geometry=geometry,
    )


# --- Input Generation for Data Fusion ---


def generate_STARS_inputs(
    tile: str,
    date_UTC: date,
    HLS_start_date: date,
    HLS_end_date: date,
    VIIRS_start_date: date,
    VIIRS_end_date: date,
    NDVI_resolution: int,
    albedo_resolution: int,
    target_resolution: int,
    NDVI_coarse_geometry: RasterGeometry,
    albedo_coarse_geometry: RasterGeometry,
    working_directory: str,
    NDVI_coarse_directory: str,
    NDVI_fine_directory: str,
    albedo_coarse_directory: str,
    albedo_fine_directory: str,
    HLS_connection: HLS2CMR,
    NDVI_VIIRS_connection: VIIRSDownloaderNDVI,
    albedo_VIIRS_connection: VIIRSDownloaderAlbedo,
    calibrate_fine: bool = False,
):
    """
    Generates and stages the necessary coarse and fine resolution input images
    for the STARS data fusion process.

    This function iterates through the VIIRS date range, retrieving and saving
    coarse NDVI and albedo images. For dates within the HLS range, it also
    retrieves and saves fine NDVI and albedo images. It can optionally
    calibrate the fine images to the coarse images.

    Args:
        tile (str): The HLS tile ID.
        date_UTC (date): The target UTC date for the L2T_STARS product.
        HLS_start_date (date): The start date for HLS data retrieval for the fusion period.
        HLS_end_date (date): The end date for HLS data retrieval for the fusion period.
        VIIRS_start_date (date): The start date for VIIRS data retrieval for the fusion period.
        VIIRS_end_date (date): The end date for VIIRS data retrieval for the fusion period.
        NDVI_resolution (int): The resolution of the coarse NDVI data.
        albedo_resolution (int): The resolution of the coarse albedo data.
        target_resolution (int): The desired output resolution of the fused product.
        NDVI_coarse_geometry (RasterGeometry): The target geometry for coarse NDVI images.
        albedo_coarse_geometry (RasterGeometry): The target geometry for coarse albedo images.
        working_directory (str): The main working directory.
        NDVI_coarse_directory (str): Directory for staging coarse NDVI images.
        NDVI_fine_directory (str): Directory for staging fine NDVI images.
        albedo_coarse_directory (str): Directory for staging coarse albedo images.
        albedo_fine_directory (str): Directory for staging fine albedo images.
        HLS_connection (HLS2CMR): An initialized HLS data connection object.
        NDVI_VIIRS_connection (VIIRSDownloaderNDVI): An initialized VIIRS NDVI downloader.
        albedo_VIIRS_connection (VIIRSDownloaderAlbedo): An initialized VIIRS albedo downloader.
        calibrate_fine (bool, optional): If True, calibrate fine images to coarse images.
                                         Defaults to False.

    Raises:
        AncillaryLatency: If coarse VIIRS data is missing within the VIIRS_GIVEUP_DAYS window.
    """
    missing_coarse_dates = set()  # Track dates where coarse data could not be generated

    # Process each day within the VIIRS data fusion window
    for processing_date in [
        get_date(dt) for dt in rrule(DAILY, dtstart=VIIRS_start_date, until=VIIRS_end_date)
    ]:
        logger.info(
            f"Preparing coarse image for STARS NDVI at {cl.place(tile)} on {cl.time(processing_date)}"
        )

        try:
            # Generate coarse NDVI image
            NDVI_coarse_image = generate_NDVI_coarse_image(
                date_UTC=processing_date,
                VIIRS_connection=NDVI_VIIRS_connection,
                geometry=NDVI_coarse_geometry,
            )

            # Define filename for coarse NDVI and save
            NDVI_coarse_filename = generate_filename(
                directory=NDVI_coarse_directory,
                variable="NDVI",
                date_UTC=processing_date,
                tile=tile,
                cell_size=NDVI_resolution,
            )
            logger.info(
                f"Saving coarse image for STARS NDVI at {cl.place(tile)} on {cl.time(processing_date)}: {NDVI_coarse_filename}"
            )
            NDVI_coarse_image.to_geotiff(NDVI_coarse_filename)

            # If the processing date is within the HLS range, generate fine NDVI
            if processing_date >= HLS_start_date:
                logger.info(
                    f"Preparing fine image for STARS NDVI at {cl.place(tile)} on {cl.time(processing_date)}"
                )
                try:
                    NDVI_fine_image = generate_NDVI_fine_image(
                        date_UTC=processing_date,
                        tile=tile,
                        HLS_connection=HLS_connection,
                    )

                    # Optionally calibrate the fine NDVI image to the coarse NDVI image
                    if calibrate_fine:
                        logger.info(
                            f"Calibrating fine image for STARS NDVI at {cl.place(tile)} on {cl.time(processing_date)}"
                        )
                        NDVI_fine_image = calibrate_fine_to_coarse(
                            NDVI_fine_image, NDVI_coarse_image
                        )

                    # Define filename for fine NDVI and save
                    NDVI_fine_filename = generate_filename(
                        directory=NDVI_fine_directory,
                        variable="NDVI",
                        date_UTC=processing_date,
                        tile=tile,
                        cell_size=target_resolution,
                    )
                    logger.info(
                        f"Saving fine image for STARS NDVI at {cl.place(tile)} on {cl.time(processing_date)}: {NDVI_fine_filename}"
                    )
                    NDVI_fine_image.to_geotiff(NDVI_fine_filename)
                except Exception:  # Catch any exception during HLS fine image generation
                    logger.info(f"HLS NDVI is not available on {processing_date}")
        except Exception as e:
            logger.exception(e)
            logger.warning(
                f"Unable to produce coarse NDVI for date {processing_date}"
            )
            missing_coarse_dates.add(processing_date)  # Add date to missing set

        logger.info(
            f"Preparing coarse image for STARS albedo at {cl.place(tile)} on {cl.time(processing_date)}"
        )
        try:
            # Generate coarse albedo image
            albedo_coarse_image = generate_albedo_coarse_image(
                date_UTC=processing_date,
                VIIRS_connection=albedo_VIIRS_connection,
                geometry=albedo_coarse_geometry,
            )

            # Define filename for coarse albedo and save
            albedo_coarse_filename = generate_filename(
                directory=albedo_coarse_directory,
                variable="albedo",
                date_UTC=processing_date,
                tile=tile,
                cell_size=albedo_resolution,
            )
            logger.info(
                f"Saving coarse image for STARS albedo at {cl.place(tile)} on {cl.time(processing_date)}: {albedo_coarse_filename}"
            )
            albedo_coarse_image.to_geotiff(albedo_coarse_filename)

            # If the processing date is within the HLS range, generate fine albedo
            if processing_date >= HLS_start_date:
                logger.info(
                    f"Preparing fine image for STARS albedo at {cl.place(tile)} on {cl.time(processing_date)}"
                )
                try:
                    albedo_fine_image = generate_albedo_fine_image(
                        date_UTC=processing_date,
                        tile=tile,
                        HLS_connection=HLS_connection,
                    )

                    # Optionally calibrate the fine albedo image to the coarse albedo image
                    if calibrate_fine:
                        logger.info(
                            f"Calibrating fine image for STARS albedo at {cl.place(tile)} on {cl.time(processing_date)}"
                        )
                        albedo_fine_image = calibrate_fine_to_coarse(
                            albedo_fine_image, albedo_coarse_image
                        )

                    # Define filename for fine albedo and save
                    albedo_fine_filename = generate_filename(
                        directory=albedo_fine_directory,
                        variable="albedo",
                        date_UTC=processing_date,
                        tile=tile,
                        cell_size=target_resolution,
                    )
                    logger.info(
                        f"Saving fine image for STARS albedo at {cl.place(tile)} on {cl.time(processing_date)}: {albedo_fine_filename}"
                    )
                    albedo_fine_image.to_geotiff(albedo_fine_filename)
                except Exception:  # Catch any exception during HLS fine image generation
                    logger.info(f"HLS albedo is not available on {processing_date}")
        except Exception as e:
            logger.exception(e)
            logger.warning(
                f"Unable to produce coarse albedo for date {processing_date}"
            )
            missing_coarse_dates.add(processing_date)  # Add date to missing set

    # Check for missing coarse dates within the give-up window
    coarse_latency_dates = [
        d
        for d in missing_coarse_dates
        if (datetime.utcnow().date() - d).days <= VIIRS_GIVEUP_DAYS
    ]

    if len(coarse_latency_dates) > 0:
        raise AncillaryLatency(
            f"Missing coarse dates within {VIIRS_GIVEUP_DAYS}-day window: "
            f"{', '.join([str(d) for d in sorted(list(coarse_latency_dates))])}"
        )


def load_prior(
    tile: str, target_resolution: int, model_directory: str, L2T_STARS_prior_filename: str
) -> Prior:
    """
    Loads a prior L2T_STARS product if available and extracts its components.

    This function attempts to load a previously generated L2T_STARS product
    to use its NDVI and albedo mean, uncertainty, and bias as prior information
    for the current data fusion run.

    Args:
        tile (str): The HLS tile ID.
        target_resolution (int): The target resolution of the L2T_STARS product.
        model_directory (str): The base directory for model state files.
        L2T_STARS_prior_filename (str): The filename of the prior L2T_STARS product (zip file).

    Returns:
        Prior: A Prior object containing paths to the prior data components
               and a flag indicating if a valid prior was loaded.
    """
    using_prior = False
    prior_date_UTC = None
    prior_NDVI_filename = None
    prior_NDVI_UQ_filename = None
    prior_NDVI_bias_filename = None
    prior_NDVI_bias_UQ_filename = None
    prior_albedo_filename = None
    prior_albedo_UQ_filename = None
    prior_albedo_bias_filename = None
    prior_albedo_bias_UQ_filename = None

    # Check if a prior L2T_STARS product is specified and exists
    if L2T_STARS_prior_filename is not None and exists(L2T_STARS_prior_filename):
        logger.info(f"Loading prior L2T STARS product: {L2T_STARS_prior_filename}")
        try:
            # Initialize L2TSTARS object from the prior product
            L2T_STARS_prior_granule = L2TSTARS(L2T_STARS_prior_filename)
            prior_date_UTC = L2T_STARS_prior_granule.date_UTC
            logger.info(f"Prior date: {cl.time(prior_date_UTC)}")

            # Extract NDVI and albedo mean and uncertainty rasters
            NDVI_prior_mean = L2T_STARS_prior_granule.NDVI
            NDVI_prior_UQ = L2T_STARS_prior_granule.NDVI_UQ
            albedo_prior_mean = L2T_STARS_prior_granule.albedo
            albedo_prior_UQ = L2T_STARS_prior_granule.albedo_UQ

            # Define the directory for storing prior model state files
            prior_tile_date_directory = generate_model_state_tile_date_directory(
                model_directory=model_directory, tile=tile, date_UTC=prior_date_UTC
            )

            # Generate and save filenames for all prior components
            prior_NDVI_filename = generate_filename(
                directory=prior_tile_date_directory,
                variable="NDVI",
                date_UTC=prior_date_UTC,
                tile=tile,
                cell_size=target_resolution,
            )
            NDVI_prior_mean.to_geotiff(prior_NDVI_filename)

            prior_NDVI_UQ_filename = generate_filename(
                directory=prior_tile_date_directory,
                variable="NDVI.UQ",
                date_UTC=prior_date_UTC,
                tile=tile,
                cell_size=target_resolution,
            )
            NDVI_prior_UQ.to_geotiff(prior_NDVI_UQ_filename)

            # Note: Prior bias files might not always exist in older products.
            # Assuming they could be extracted from L2T_STARS_prior_granule if present.
            # For simplicity, these are placeholders and would need proper extraction logic
            # from the L2TSTARS granule if they are actual components.
            # For now, we set them based on `generate_filename` only if a bias layer is retrieved.
            # If the bias layers are not explicitly part of `L2TSTARS` object, these will be None
            # unless written explicitly during prior creation.
            prior_NDVI_bias_filename = generate_filename(
                directory=prior_tile_date_directory,
                variable="NDVI.bias",
                date_UTC=prior_date_UTC,
                tile=tile,
                cell_size=target_resolution,
            )
            # Assuming L2T_STARS_prior_granule has a .NDVI_bias attribute
            if hasattr(L2T_STARS_prior_granule, "NDVI_bias") and L2T_STARS_prior_granule.NDVI_bias is not None:
                L2T_STARS_prior_granule.NDVI_bias.to_geotiff(prior_NDVI_bias_filename)
            else:
                prior_NDVI_bias_filename = None # Set to None if not available

            prior_NDVI_bias_UQ_filename = generate_filename(
                directory=prior_tile_date_directory,
                variable="NDVI.bias.UQ",
                date_UTC=prior_date_UTC,
                tile=tile,
                cell_size=target_resolution,
            )
            # Assuming L2T_STARS_prior_granule has a .NDVI_bias_UQ attribute
            if hasattr(L2T_STARS_prior_granule, "NDVI_bias_UQ") and L2T_STARS_prior_granule.NDVI_bias_UQ is not None:
                L2T_STARS_prior_granule.NDVI_bias_UQ.to_geotiff(prior_NDVI_bias_UQ_filename)
            else:
                prior_NDVI_bias_UQ_filename = None # Set to None if not available


            prior_albedo_filename = generate_filename(
                directory=prior_tile_date_directory,
                variable="albedo",
                date_UTC=prior_date_UTC,
                tile=tile,
                cell_size=target_resolution,
            )
            albedo_prior_mean.to_geotiff(prior_albedo_filename)

            prior_albedo_UQ_filename = generate_filename(
                directory=prior_tile_date_directory,
                variable="albedo.UQ",
                date_UTC=prior_date_UTC,
                tile=tile,
                cell_size=target_resolution,
            )
            albedo_prior_UQ.to_geotiff(prior_albedo_UQ_filename)

            prior_albedo_bias_filename = generate_filename(
                directory=prior_tile_date_directory,
                variable="albedo.bias",
                date_UTC=prior_date_UTC,
                tile=tile,
                cell_size=target_resolution,
            )
            # Assuming L2T_STARS_prior_granule has a .albedo_bias attribute
            if hasattr(L2T_STARS_prior_granule, "albedo_bias") and L2T_STARS_prior_granule.albedo_bias is not None:
                L2T_STARS_prior_granule.albedo_bias.to_geotiff(prior_albedo_bias_filename)
            else:
                prior_albedo_bias_filename = None # Set to None if not available

            prior_albedo_bias_UQ_filename = generate_filename(
                directory=prior_tile_date_directory,
                variable="albedo.bias.UQ",
                date_UTC=prior_date_UTC,
                tile=tile,
                cell_size=target_resolution,
            )
            # Assuming L2T_STARS_prior_granule has a .albedo_bias_UQ attribute
            if hasattr(L2T_STARS_prior_granule, "albedo_bias_UQ") and L2T_STARS_prior_granule.albedo_bias_UQ is not None:
                L2T_STARS_prior_granule.albedo_bias_UQ.to_geotiff(prior_albedo_bias_UQ_filename)
            else:
                prior_albedo_bias_UQ_filename = None # Set to None if not available


            using_prior = True # Mark that a prior was successfully loaded
        except Exception as e:
            logger.warning(f"Could not load prior L2T STARS product: {L2T_STARS_prior_filename}. Reason: {e}")
            using_prior = False # Ensure using_prior is False if loading fails

    # Verify that all expected prior files exist, otherwise disable using_prior
    # This ensures a partial prior (e.g., only NDVI but not albedo) isn't used
    if prior_NDVI_filename is not None and exists(prior_NDVI_filename):
        logger.info(f"Prior NDVI ready: {prior_NDVI_filename}")
    else:
        logger.info(f"Prior NDVI not found: {prior_NDVI_filename}")
        using_prior = False

    if prior_NDVI_UQ_filename is not None and exists(prior_NDVI_UQ_filename):
        logger.info(f"Prior NDVI UQ ready: {prior_NDVI_UQ_filename}")
    else:
        logger.info(f"Prior NDVI UQ not found: {prior_NDVI_UQ_filename}")
        using_prior = False

    if prior_NDVI_bias_filename is not None and exists(prior_NDVI_bias_filename):
        logger.info(f"Prior NDVI bias ready: {prior_NDVI_bias_filename}")
    else:
        logger.info(f"Prior NDVI bias not found: {prior_NDVI_bias_filename}")
        using_prior = False

    if prior_NDVI_bias_UQ_filename is not None and exists(prior_NDVI_bias_UQ_filename):
        logger.info(f"Prior NDVI bias UQ ready: {prior_NDVI_bias_UQ_filename}")
    else:
        logger.info(f"Prior NDVI bias UQ not found: {prior_NDVI_bias_UQ_filename}")
        using_prior = False

    if prior_albedo_filename is not None and exists(prior_albedo_filename):
        logger.info(f"Prior albedo ready: {prior_albedo_filename}")
    else:
        logger.info(f"Prior albedo not found: {prior_albedo_filename}")
        using_prior = False

    if prior_albedo_UQ_filename is not None and exists(prior_albedo_UQ_filename):
        logger.info(f"Prior albedo UQ ready: {prior_albedo_UQ_filename}")
    else:
        logger.info(f"Prior albedo UQ not found: {prior_albedo_UQ_filename}")
        using_prior = False

    if prior_albedo_bias_filename is not None and exists(prior_albedo_bias_filename):
        logger.info(f"Prior albedo bias ready: {prior_albedo_bias_filename}")
    else:
        logger.info(f"Prior albedo bias not found: {prior_albedo_bias_filename}")
        using_prior = False

    if prior_albedo_bias_UQ_filename is not None and exists(
        prior_albedo_bias_UQ_filename
    ):
        logger.info(f"Prior albedo bias UQ ready: {prior_albedo_bias_UQ_filename}")
    else:
        logger.info(f"Prior albedo bias UQ not found: {prior_albedo_bias_UQ_filename}")
        using_prior = False
    
    # Final check: if any of the critical prior files are missing, do not use prior
    if not all([prior_NDVI_filename, prior_NDVI_UQ_filename, prior_albedo_filename, prior_albedo_UQ_filename]):
        logger.warning("One or more required prior (mean/UQ) files are missing. Prior will not be used.")
        using_prior = False
        prior_NDVI_filename = None
        prior_NDVI_UQ_filename = None
        prior_albedo_filename = None
        prior_albedo_UQ_filename = None

    # Create and return the Prior object
    prior = Prior(
        using_prior=using_prior,
        prior_date_UTC=prior_date_UTC,
        L2T_STARS_prior_filename=L2T_STARS_prior_filename,
        prior_NDVI_filename=prior_NDVI_filename,
        prior_NDVI_UQ_filename=prior_NDVI_UQ_filename,
        prior_NDVI_flag_filename=prior_NDVI_flag_filename,
        prior_NDVI_bias_filename=prior_NDVI_bias_filename,
        prior_NDVI_bias_UQ_filename=prior_NDVI_bias_UQ_filename,
        prior_albedo_filename=prior_albedo_filename,
        prior_albedo_UQ_filename=prior_albedo_UQ_filename,
        prior_albedo_flag_filename=prior_albedo_flag_filename,
        prior_albedo_bias_filename=prior_albedo_bias_filename,
        prior_albedo_bias_UQ_filename=prior_albedo_bias_UQ_filename,
    )

    return prior


def process_STARS_product(
    tile: str,
    date_UTC: date,
    time_UTC: datetime,
    build: str,
    product_counter: int,
    HLS_start_date: date,
    HLS_end_date: date,
    VIIRS_start_date: date,
    VIIRS_end_date: date,
    NDVI_resolution: int,
    albedo_resolution: int,
    target_resolution: int,
    working_directory: str,
    model_directory: str,
    input_staging_directory: str,
    L2T_STARS_granule_directory: str,
    L2T_STARS_zip_filename: str,
    L2T_STARS_browse_filename: str,
    metadata: dict,
    prior: Prior,
    HLS_connection: HLS2CMR,
    NDVI_VIIRS_connection: VIIRSDownloaderNDVI,
    albedo_VIIRS_connection: VIIRSDownloaderAlbedo,
    using_prior: bool = False,
    calibrate_fine: bool = False,
    remove_input_staging: bool = True,
    remove_prior: bool = True,
    remove_posterior: bool = True,
    threads: Union[int, str] = "auto",
    num_workers: int = 4,
):
    """
    Orchestrates the generation of the L2T_STARS product for a given tile and date.

    This function handles the staging of input data, execution of the Julia data
    fusion model for both NDVI and albedo, and the final assembly, metadata
    writing, and archiving of the L2T_STARS product. It also manages cleanup
    of intermediate and prior files.

    Args:
        tile (str): The HLS tile ID.
        date_UTC (date): The UTC date for the current L2T_STARS product.
        time_UTC (datetime): The UTC time for the current L2T_STARS product.
        build (str): The build ID of the PGE.
        product_counter (int): The product counter for the current run.
        HLS_start_date (date): The start date for HLS data used in fusion.
        HLS_end_date (date): The end date for HLS data used in fusion.
        VIIRS_start_date (date): The start date for VIIRS data used in fusion.
        VIIRS_end_date (date): The end date for VIIRS data used in fusion.
        NDVI_resolution (int): The resolution of the coarse NDVI data.
        albedo_resolution (int): The resolution of the coarse albedo data.
        target_resolution (int): The desired output resolution of the fused product.
        working_directory (str): The main working directory.
        model_directory (str): Directory for model state files (priors, posteriors).
        input_staging_directory (str): Directory for temporary input images for Julia.
        L2T_STARS_granule_directory (str): Temporary directory for the unzipped product.
        L2T_STARS_zip_filename (str): Final path for the zipped L2T_STARS product.
        L2T_STARS_browse_filename (str): Final path for the browse image.
        metadata (dict): Dictionary containing product metadata.
        prior (Prior): An object containing information about the prior product.
        HLS_connection (HLS2CMR): An initialized HLS data connection object.
        NDVI_VIIRS_connection (VIIRSDownloaderNDVI): An initialized VIIRS NDVI downloader.
        albedo_VIIRS_connection (VIIRSDownloaderAlbedo): An initialized VIIRS albedo downloader.
        using_prior (bool, optional): If True, use the prior product in fusion. Defaults to False.
        calibrate_fine (bool, optional): If True, calibrate fine images to coarse images.
                                         Defaults to False.
        remove_input_staging (bool, optional): If True, remove the input staging directory
                                                after processing. Defaults to True.
        remove_prior (bool, optional): If True, remove prior intermediate files after use.
                                       Defaults to True.
        remove_posterior (bool, optional): If True, remove posterior intermediate files after
                                           product generation. Defaults to True.
        threads (Union[int, str], optional): Number of Julia threads to use, or "auto".
                                            Defaults to "auto".
        num_workers (int, str): Number of Julia workers for distributed processing.
                                     Defaults to 4.

    Raises:
        BlankOutput: If any of the final fused output rasters (NDVI, albedo, UQ, flag) are empty.
    """
    # Get the target geometries for coarse NDVI and albedo based on the HLS grid
    NDVI_coarse_geometry = HLS_connection.grid(tile=tile, cell_size=NDVI_resolution)
    albedo_coarse_geometry = HLS_connection.grid(tile=tile, cell_size=albedo_resolution)

    logger.info(f"Processing the L2T_STARS product at tile {cl.place(tile)} for date {cl.time(date_UTC)}")

    # Define and create input staging directories for coarse and fine NDVI/albedo
    NDVI_coarse_directory = generate_NDVI_coarse_directory(
        input_staging_directory=input_staging_directory, tile=tile
    )
    logger.info(f"Staging coarse NDVI images: {cl.dir(NDVI_coarse_directory)}")

    NDVI_fine_directory = generate_NDVI_fine_directory(
        input_staging_directory=input_staging_directory, tile=tile
    )
    logger.info(f"Staging fine NDVI images: {cl.dir(NDVI_fine_directory)}")

    albedo_coarse_directory = generate_albedo_coarse_directory(
        input_staging_directory=input_staging_directory, tile=tile
    )
    logger.info(f"Staging coarse albedo images: {cl.dir(albedo_coarse_directory)}")

    albedo_fine_directory = generate_albedo_fine_directory(
        input_staging_directory=input_staging_directory, tile=tile
    )
    logger.info(f"Staging fine albedo images: {cl.dir(albedo_fine_directory)}")

    # Define and create the directory for storing posterior model state files
    posterior_tile_date_directory = generate_model_state_tile_date_directory(
        model_directory=model_directory, tile=tile, date_UTC=date_UTC
    )
    logger.info(f"Posterior directory: {cl.dir(posterior_tile_date_directory)}")

    # Generate the actual input raster files (coarse and fine images)
    generate_STARS_inputs(
        tile=tile,
        date_UTC=date_UTC,
        HLS_start_date=HLS_start_date,
        HLS_end_date=HLS_end_date,
        VIIRS_start_date=VIIRS_start_date,
        VIIRS_end_date=VIIRS_end_date,
        NDVI_resolution=NDVI_resolution,
        albedo_resolution=albedo_resolution,
        target_resolution=target_resolution,
        NDVI_coarse_geometry=NDVI_coarse_geometry,
        albedo_coarse_geometry=albedo_coarse_geometry,
        working_directory=working_directory,
        NDVI_coarse_directory=NDVI_coarse_directory,
        NDVI_fine_directory=NDVI_fine_directory,
        albedo_coarse_directory=albedo_coarse_directory,
        albedo_fine_directory=albedo_fine_directory,
        HLS_connection=HLS_connection,
        NDVI_VIIRS_connection=NDVI_VIIRS_connection,
        albedo_VIIRS_connection=albedo_VIIRS_connection,
        calibrate_fine=calibrate_fine,
    )

    # --- Process NDVI Data Fusion ---
    # Define output filenames for NDVI posterior products
    posterior_NDVI_filename = generate_filename(
        directory=posterior_tile_date_directory,
        variable="NDVI",
        date_UTC=date_UTC,
        tile=tile,
        cell_size=target_resolution,
    )
    logger.info(f"Posterior NDVI file: {cl.file(posterior_NDVI_filename)}")

    posterior_NDVI_UQ_filename = generate_filename(
        directory=posterior_tile_date_directory,
        variable="NDVI.UQ",
        date_UTC=date_UTC,
        tile=tile,
        cell_size=target_resolution,
    )
    logger.info(f"Posterior NDVI UQ file: {cl.file(posterior_NDVI_UQ_filename)}")

    posterior_NDVI_flag_filename = generate_filename(
        directory=posterior_tile_date_directory,
        variable="NDVI.flag",
        date_UTC=date_UTC,
        tile=tile,
        cell_size=target_resolution,
    )
    logger.info(f"Posterior NDVI flag file: {cl.file(posterior_NDVI_flag_filename)}")

    posterior_NDVI_bias_filename = generate_filename(
        directory=posterior_tile_date_directory,
        variable="NDVI.bias",
        date_UTC=date_UTC,
        tile=tile,
        cell_size=target_resolution,
    )
    logger.info(f"Posterior NDVI bias file: {cl.file(posterior_NDVI_bias_filename)}")

    posterior_NDVI_bias_UQ_filename = generate_filename(
        directory=posterior_tile_date_directory,
        variable="NDVI.bias.UQ",
        date_UTC=date_UTC,
        tile=tile,
        cell_size=target_resolution,
    )
    logger.info(f"Posterior NDVI bias UQ file: {cl.file(posterior_NDVI_bias_UQ_filename)}")

    # Run Julia data fusion for NDVI, conditionally including prior data
    if using_prior:
        logger.info("Running Julia data fusion for NDVI with prior data.")
        process_julia_data_fusion(
            tile=tile,
            coarse_cell_size=NDVI_resolution,
            fine_cell_size=target_resolution,
            VIIRS_start_date=VIIRS_start_date,
            VIIRS_end_date=VIIRS_end_date,
            HLS_start_date=HLS_start_date,
            HLS_end_date=HLS_end_date,
            coarse_directory=NDVI_coarse_directory,
            fine_directory=NDVI_fine_directory,
            posterior_filename=posterior_NDVI_filename,
            posterior_UQ_filename=posterior_NDVI_UQ_filename,
            posterior_flag_filename=posterior_NDVI_flag_filename,
            posterior_bias_filename=posterior_NDVI_bias_filename,
            posterior_bias_UQ_filename=posterior_NDVI_bias_UQ_filename,
            prior_filename=prior.prior_NDVI_filename,
            prior_UQ_filename=prior.prior_NDVI_UQ_filename,
            prior_bias_filename=prior.prior_NDVI_bias_filename,
            prior_bias_UQ_filename=prior.prior_NDVI_bias_UQ_filename,
            threads=threads,
            num_workers=num_workers,
        )
    else:
        logger.info("Running Julia data fusion for NDVI without prior data.")
        process_julia_data_fusion(
            tile=tile,
            coarse_cell_size=NDVI_resolution,
            fine_cell_size=target_resolution,
            VIIRS_start_date=VIIRS_start_date,
            VIIRS_end_date=VIIRS_end_date,
            HLS_start_date=HLS_start_date,
            HLS_end_date=HLS_end_date,
            coarse_directory=NDVI_coarse_directory,
            fine_directory=NDVI_fine_directory,
            posterior_filename=posterior_NDVI_filename,
            posterior_UQ_filename=posterior_NDVI_UQ_filename,
            posterior_flag_filename=posterior_NDVI_flag_filename,
            posterior_bias_filename=posterior_NDVI_bias_filename,
            posterior_bias_UQ_filename=posterior_NDVI_bias_UQ_filename,
            threads=threads,
            num_workers=num_workers,
        )

    # Open the resulting NDVI rasters
    NDVI = Raster.open(posterior_NDVI_filename)
    NDVI_UQ = Raster.open(posterior_NDVI_UQ_filename)
    NDVI_flag = Raster.open(posterior_NDVI_flag_filename)

    # --- Process Albedo Data Fusion ---
    # Define output filenames for albedo posterior products
    posterior_albedo_filename = generate_filename(
        directory=posterior_tile_date_directory,
        variable="albedo",
        date_UTC=date_UTC,
        tile=tile,
        cell_size=target_resolution,
    )
    logger.info(f"Posterior albedo file: {cl.file(posterior_albedo_filename)}")

    posterior_albedo_UQ_filename = generate_filename(
        directory=posterior_tile_date_directory,
        variable="albedo.UQ",
        date_UTC=date_UTC,
        tile=tile,
        cell_size=target_resolution,
    )
    logger.info(f"Posterior albedo UQ file: {cl.file(posterior_albedo_UQ_filename)}")

    posterior_albedo_flag_filename = generate_filename(
        directory=posterior_tile_date_directory,
        variable="albedo.flag",
        date_UTC=date_UTC,
        tile=tile,
        cell_size=target_resolution,
    )
    logger.info(f"Posterior albedo flag file: {cl.file(posterior_albedo_flag_filename)}")

    posterior_albedo_bias_filename = generate_filename(
        directory=posterior_tile_date_directory,
        variable="albedo.bias",
        date_UTC=date_UTC,
        tile=tile,
        cell_size=target_resolution,
    )
    logger.info(f"Posterior albedo bias file: {cl.file(posterior_albedo_bias_filename)}")

    posterior_albedo_bias_UQ_filename = generate_filename(
        directory=posterior_tile_date_directory,
        variable="albedo.bias.UQ",
        date_UTC=date_UTC,
        tile=tile,
        cell_size=target_resolution,
    )
    logger.info(f"Posterior albedo bias UQ file: {cl.file(posterior_albedo_bias_UQ_filename)}")

    # Run Julia data fusion for albedo, conditionally including prior data
    if using_prior:
        logger.info("Running Julia data fusion for albedo with prior data.")
        process_julia_data_fusion(
            tile=tile,
            coarse_cell_size=albedo_resolution,
            fine_cell_size=target_resolution,
            VIIRS_start_date=VIIRS_start_date,
            VIIRS_end_date=VIIRS_end_date,
            HLS_start_date=HLS_start_date,
            HLS_end_date=HLS_end_date,
            coarse_directory=albedo_coarse_directory,
            fine_directory=albedo_fine_directory,
            posterior_filename=posterior_albedo_filename,
            posterior_UQ_filename=posterior_albedo_UQ_filename,
            posterior_flag_filename=posterior_albedo_flag_filename,
            posterior_bias_filename=posterior_albedo_bias_filename,
            posterior_bias_UQ_filename=posterior_albedo_bias_UQ_filename,
            prior_filename=prior.prior_albedo_filename,
            prior_UQ_filename=prior.prior_albedo_UQ_filename,
            prior_bias_filename=prior.prior_albedo_bias_filename,
            prior_bias_UQ_filename=prior.prior_albedo_bias_UQ_filename,
            threads=threads,
            num_workers=num_workers,
        )
    else:
        logger.info("Running Julia data fusion for albedo without prior data.")
        process_julia_data_fusion(
            tile=tile,
            coarse_cell_size=albedo_resolution,
            fine_cell_size=target_resolution,
            VIIRS_start_date=VIIRS_start_date,
            VIIRS_end_date=VIIRS_end_date,
            HLS_start_date=HLS_start_date,
            HLS_end_date=HLS_end_date,
            coarse_directory=albedo_coarse_directory,
            fine_directory=albedo_fine_directory,
            posterior_filename=posterior_albedo_filename,
            posterior_UQ_filename=posterior_albedo_UQ_filename,
            posterior_flag_filename=posterior_albedo_flag_filename,
            posterior_bias_filename=posterior_albedo_bias_filename,
            posterior_bias_UQ_filename=posterior_albedo_bias_UQ_filename,
            threads=threads,
            num_workers=num_workers,
        )

    # Open the resulting albedo rasters
    albedo = Raster.open(posterior_albedo_filename)
    albedo_UQ = Raster.open(posterior_albedo_UQ_filename)
    albedo_flag = Raster.open(posterior_albedo_flag_filename)

    # --- Validate Output and Create Final Product ---
    # Check if the output rasters are valid (not None, indicating a problem during Julia processing)
    if NDVI is None:
        raise BlankOutput("Unable to generate STARS NDVI")
    if NDVI_UQ is None:
        raise BlankOutput("Unable to generate STARS NDVI UQ")
    if NDVI_flag is None:
        raise BlankOutput("Unable to generate STARS NDVI flag")
    if albedo is None:
        raise BlankOutput("Unable to generate STARS albedo")
    if albedo_UQ is None:
        raise BlankOutput("Unable to generate STARS albedo UQ")
    if albedo_flag is None:
        raise BlankOutput("Unable to generate STARS albedo flag")

    # Initialize the L2TSTARS granule object for the current product
    granule = L2TSTARS(
        product_location=L2T_STARS_granule_directory,
        tile=tile,
        time_UTC=time_UTC,
        build=build,
        process_count=product_counter,
    )

    # Add the generated layers to the granule object
    granule.add_layer("NDVI", NDVI, cmap=NDVI_COLORMAP)
    granule.add_layer("NDVI-UQ", NDVI_UQ, cmap="jet")
    granule.add_layer("NDVI-flag", NDVI_flag, cmap="jet")
    granule.add_layer("albedo", albedo, cmap=ALBEDO_COLORMAP)
    granule.add_layer("albedo-UQ", albedo_UQ, cmap="jet")
    granule.add_layer("albedo-flag", albedo_flag, cmap="jet")

    # Update metadata and write to the granule
    metadata["StandardMetadata"]["LocalGranuleID"] = basename(L2T_STARS_zip_filename)
    metadata["StandardMetadata"]["SISName"] = "Level 2 STARS Product Specification Document"
    granule.write_metadata(metadata)

    # Write the zipped product and browse image
    logger.info(f"Writing L2T STARS product zip: {cl.file(L2T_STARS_zip_filename)}")
    granule.write_zip(L2T_STARS_zip_filename)
    logger.info(f"Writing L2T STARS browse image: {cl.file(L2T_STARS_browse_filename)}")
    granule.write_browse_image(PNG_filename=L2T_STARS_browse_filename)
    logger.info(
        f"Removing L2T STARS tile granule directory: {cl.dir(L2T_STARS_granule_directory)}"
    )
    shutil.rmtree(L2T_STARS_granule_directory)

    # Re-check and regenerate browse image if it somehow didn't generate (e.g. if the granule dir was already deleted)
    if not exists(L2T_STARS_browse_filename):
        logger.info(
            f"Browse image not found after initial creation attempt. Regenerating: {cl.file(L2T_STARS_browse_filename)}"
        )
        # Re-load granule from zip to create browse image if necessary
        granule_for_browse = L2TSTARS(L2T_STARS_zip_filename)
        granule_for_browse.write_browse_image(PNG_filename=L2T_STARS_browse_filename)

    # Re-write posterior files (often done to ensure proper compression/color maps after processing)
    # This step might be redundant if Julia already writes them correctly, but ensures consistency.
    logger.info(f"Re-writing posterior NDVI: {posterior_NDVI_filename}")
    Raster.open(posterior_NDVI_filename, cmap=NDVI_COLORMAP).to_geotiff(
        posterior_NDVI_filename
    )
    logger.info(f"Re-writing posterior NDVI UQ: {posterior_NDVI_UQ_filename}")
    Raster.open(posterior_NDVI_UQ_filename, cmap="jet").to_geotiff(
        posterior_NDVI_UQ_filename
    )
    logger.info(f"Re-writing posterior NDVI flag: {posterior_NDVI_flag_filename}")
    Raster.open(posterior_NDVI_flag_filename, cmap="jet").to_geotiff(
        posterior_NDVI_flag_filename
    )
    logger.info(f"Re-writing posterior NDVI bias: {posterior_NDVI_bias_filename}")
    Raster.open(posterior_NDVI_bias_filename, cmap=NDVI_COLORMAP).to_geotiff(
        posterior_NDVI_bias_filename
    )
    logger.info(f"Re-writing posterior NDVI bias UQ: {posterior_NDVI_bias_UQ_filename}")
    Raster.open(posterior_NDVI_bias_UQ_filename, cmap=NDVI_COLORMAP).to_geotiff(
        posterior_NDVI_bias_UQ_filename
    )

    logger.info(f"Re-writing posterior albedo: {posterior_albedo_filename}")
    Raster.open(posterior_albedo_filename, cmap=ALBEDO_COLORMAP).to_geotiff(
        posterior_albedo_filename
    )
    logger.info(f"Re-writing posterior albedo UQ: {posterior_albedo_UQ_filename}")
    Raster.open(posterior_albedo_UQ_filename, cmap="jet").to_geotiff(
        posterior_albedo_UQ_filename
    )
    logger.info(f"Re-writing posterior albedo flag: {posterior_albedo_flag_filename}")
    Raster.open(posterior_albedo_flag_filename, cmap="jet").to_geotiff(
        posterior_albedo_flag_filename
    )
    logger.info(f"Re-writing posterior albedo bias: {posterior_albedo_bias_filename}")
    Raster.open(posterior_albedo_bias_filename, cmap=ALBEDO_COLORMAP).to_geotiff(
        posterior_albedo_bias_filename
    )
    logger.info(f"Re-writing posterior albedo bias UQ: {posterior_albedo_bias_UQ_filename}")
    Raster.open(posterior_albedo_bias_UQ_filename, cmap=ALBEDO_COLORMAP).to_geotiff(
        posterior_albedo_bias_UQ_UQ_filename
    )

    # --- Cleanup ---
    if remove_input_staging:
        if exists(input_staging_directory):
            logger.info(f"Removing input staging directory: {cl.dir(input_staging_directory)}")
            shutil.rmtree(input_staging_directory)

    if using_prior and remove_prior:
        # Remove prior intermediate files only if they exist
        prior_files = [
            prior.prior_NDVI_filename,
            prior.prior_NDVI_UQ_filename,
            prior.prior_NDVI_bias_filename,
            prior.prior_NDVI_bias_UQ_filename,
            prior.prior_albedo_filename,
            prior.prior_albedo_UQ_filename,
            prior.prior_albedo_bias_filename,
            prior.prior_albedo_bias_UQ_filename,
        ]
        for f in prior_files:
            if f is not None and exists(f):
                logger.info(f"Removing prior file: {cl.file(f)}")
                remove(f)

    if remove_posterior:
        # Remove posterior intermediate files only if they exist
        posterior_files = [
            posterior_NDVI_filename,
            posterior_NDVI_UQ_filename,
            posterior_NDVI_flag_filename,
            posterior_NDVI_bias_filename,
            posterior_NDVI_bias_UQ_filename,
            posterior_albedo_filename,
            posterior_albedo_UQ_filename,
            posterior_albedo_flag_filename,
            posterior_albedo_bias_filename,
            posterior_albedo_bias_UQ_filename,
        ]
        for f in posterior_files:
            if f is not None and exists(f):
                logger.info(f"Removing posterior file: {cl.file(f)}")
                remove(f)


# --- Main PGE Logic ---


def L2T_STARS(
    runconfig_filename: str,
    date_UTC: Union[date, str] = None,
    spinup_days: int = DEFAULT_SPINUP_DAYS,
    target_resolution: int = DEFAULT_TARGET_RESOLUTION,
    NDVI_resolution: int = DEFAULT_NDVI_RESOLUTION,
    albedo_resolution: int = DEFAULT_ALBEDO_RESOLUTION,
    use_VNP43NRT: bool = DEFAULT_USE_VNP43NRT,
    calibrate_fine: bool = DEFAULT_CALIBRATE_FINE,
    sources_only: bool = False,
    remove_input_staging: bool = True,
    remove_prior: bool = True,
    remove_posterior: bool = True,
    threads: Union[int, str] = "auto",
    num_workers: int = 4,
) -> int:
    """
    ECOSTRESS Collection 3 L2T_STARS PGE (Product Generation Executive).

    This function serves as the main entry point for the L2T_STARS processing.
    It orchestrates the entire workflow, including reading the run-config,
    connecting to data servers, retrieving source data, performing data fusion
    (via Julia subprocess), generating the final product, and handling cleanup.

    Args:
        runconfig_filename (str): Path to the XML run-configuration file.
        date_UTC (Union[date, str], optional): The target UTC date for product generation.
                                              If None, it's derived from the input L2T LSTE granule.
        spinup_days (int, optional): Number of days for the VIIRS time-series spin-up.
                                     Defaults to DEFAULT_SPINUP_DAYS (7).
        target_resolution (int, optional): The desired output resolution in meters.
                                           Defaults to DEFAULT_TARGET_RESOLUTION (70).
        NDVI_resolution (int, optional): The resolution of the coarse NDVI data.
                                         Defaults to DEFAULT_NDVI_RESOLUTION (490).
        albedo_resolution (int, optional): The resolution of the coarse albedo data.
                                           Defaults to DEFAULT_ALBEDO_RESOLUTION (980).
        use_VNP43NRT (bool, optional): If True, use VNP43NRT for VIIRS products.
                                       If False, use VNP43IA4 (NDVI) and VNP43MA3 (Albedo).
                                       Defaults to DEFAULT_USE_VNP43NRT (True).
        calibrate_fine (bool, optional): If True, calibrate fine resolution HLS data to
                                         coarse resolution VIIRS data. Defaults to DEFAULT_CALIBRATE_FINE (False).
        sources_only (bool, optional): If True, only retrieve source data and exit,
                                       without performing data fusion. Defaults to False.
        remove_input_staging (bool, optional): If True, remove the input staging directory
                                                after processing. Defaults to True.
        remove_prior (bool, optional): If True, remove prior intermediate files after use.
                                       Defaults to True.
        remove_posterior (bool, optional): If True, remove posterior intermediate files after
                                           product generation. Defaults to True.
        threads (Union[int, str], optional): Number of Julia threads to use, or "auto".
                                            Defaults to "auto".
        num_workers (int, optional): Number of Julia workers for distributed processing.
                                     Defaults to 4.

    Returns:
        int: An exit code indicating the success or failure of the PGE execution.
             (e.g., SUCCESS_EXIT_CODE, ANCILLARY_SERVER_UNREACHABLE, DOWNLOAD_FAILED, etc.)
    """
    exit_code = SUCCESS_EXIT_CODE  # Initialize exit code to success

    try:
        # Load and parse the run-configuration file
        runconfig = L2TSTARSConfig(runconfig_filename)

        # Configure logging with the specified log filename from runconfig
        working_directory = runconfig.working_directory
        granule_ID = runconfig.granule_ID
        log_filename = join(working_directory, "log", f"{granule_ID}.log")
        cl.configure(filename=log_filename)  # Reconfigure logger with the specific log file

        logger.info(f"L2T_STARS PGE ({cl.val(PGEVersion)})")
        logger.info(f"L2T_STARS run-config: {cl.file(runconfig_filename)}")
        logger.info(f"Granule ID: {cl.val(granule_ID)}")

        # Extract paths from the run-config
        L2T_STARS_granule_directory = runconfig.L2T_STARS_granule_directory
        logger.info(f"Granule directory: {cl.dir(L2T_STARS_granule_directory)}")
        L2T_STARS_zip_filename = runconfig.L2T_STARS_zip_filename
        logger.info(f"Zip filename: {cl.file(L2T_STARS_zip_filename)}")
        L2T_STARS_browse_filename = runconfig.L2T_STARS_browse_filename
        logger.info(f"Browse filename: " + cl.file(L2T_STARS_browse_filename))

        # Check if the final product already exists to avoid reprocessing
        if exists(L2T_STARS_zip_filename) and exists(L2T_STARS_browse_filename):
            logger.info(f"Found existing L2T STARS file: {L2T_STARS_zip_filename}")
            logger.info(f"Found existing L2T STARS preview: {L2T_STARS_browse_filename}")
            return SUCCESS_EXIT_CODE

        logger.info(f"Working directory: {cl.dir(working_directory)}")
        logger.info(f"Log file: {cl.file(log_filename)}")

        input_staging_directory = join(working_directory, "input_staging")
        logger.info(f"Input staging directory: {cl.dir(input_staging_directory)}")

        sources_directory = runconfig.sources_directory
        logger.info(f"Source directory: {cl.dir(sources_directory)}")
        indices_directory = runconfig.indices_directory
        logger.info(f"Indices directory: {cl.dir(indices_directory)}")
        model_directory = runconfig.model_directory
        logger.info(f"Model directory: {cl.dir(model_directory)}")
        output_directory = runconfig.output_directory
        logger.info(f"Output directory: {cl.dir(output_directory)}")
        tile = runconfig.tile
        logger.info(f"Tile: {cl.val(tile)}")
        build = runconfig.build
        logger.info(f"Build: {cl.val(build)}")
        product_counter = runconfig.product_counter
        logger.info(f"Product counter: {cl.val(product_counter)}")
        L2T_LSTE_filename = runconfig.L2T_LSTE_filename
        logger.info(f"Input L2T LSTE file: {cl.file(L2T_LSTE_filename)}")

        # Validate existence of input L2T LSTE file
        if not exists(L2T_LSTE_filename):
            raise InputFilesInaccessible(
                f"L2T LSTE file does not exist: {L2T_LSTE_filename}"
            )

        # Load the L2T_LSTE granule to get geometry and base metadata
        l2t_granule = L2TLSTE(L2T_LSTE_filename)
        geometry = l2t_granule.geometry
        metadata = l2t_granule.metadata_dict
        metadata["StandardMetadata"]["PGEName"] = "L2T_STARS"

        # Update product names in metadata
        short_name = L2T_STARS_SHORT_NAME
        logger.info(f"L2T STARS short name: {cl.val(short_name)}")
        metadata["StandardMetadata"]["ShortName"] = short_name

        long_name = L2T_STARS_LONG_NAME
        logger.info(f"L2T STARS long name: {cl.val(long_name)}")
        metadata["StandardMetadata"]["LongName"] = long_name

        # Update ancillary input pointers in metadata and remove irrelevant sections
        metadata["StandardMetadata"]["AncillaryInputPointer"] = "HLS,VIIRS"
        if "ProductMetadata" in metadata:
            metadata["ProductMetadata"].pop("AncillaryNWP", None)  # Safe removal
            metadata["ProductMetadata"].pop("NWPSource", None)

        # Determine the target date for processing
        time_UTC = l2t_granule.time_UTC
        logger.info(f"ECOSTRESS overpass time: {cl.time(f'{time_UTC:%Y-%m-%d %H:%M:%S} UTC')}")

        if date_UTC is None:
            # Use date from L2T granule if not provided via command line
            date_UTC = l2t_granule.date_UTC
            logger.info(f"ECOSTRESS overpass date: {cl.time(f'{date_UTC:%Y-%m-%d} UTC')}")
        else:
            logger.warning(f"Over-riding target date from command line to: {date_UTC}")
            if isinstance(date_UTC, str):
                date_UTC = parser.parse(date_UTC).date()

        # TODO: Add a check if the L2T LSTE granule is day-time and halt L2T STARS run if it's not.
        # This is a critical step to ensure valid scientific output.

        # Load prior data if specified in the run-config
        L2T_STARS_prior_filename = runconfig.L2T_STARS_prior_filename
        prior = load_prior(
            tile=tile,
            target_resolution=target_resolution,
            model_directory=model_directory,
            L2T_STARS_prior_filename=L2T_STARS_prior_filename,
        )
        using_prior = prior.using_prior
        prior_date_UTC = prior.prior_date_UTC

        # Define various product and download directories
        products_directory = join(working_directory, DEFAULT_STARS_PRODUCTS_DIRECTORY)
        logger.info(f"STARS products directory: {cl.dir(products_directory)}")
        HLS_download_directory = join(sources_directory, DEFAULT_HLS_DOWNLOAD_DIRECTORY)
        logger.info(f"HLS download directory: {cl.dir(HLS_download_directory)}")
        HLS_products_directory = join(sources_directory, DEFAULT_HLS_PRODUCTS_DIRECTORY)
        logger.info(f"HLS products directory: {cl.dir(HLS_products_directory)}")
        VIIRS_download_directory = join(sources_directory, DEFAULT_VIIRS_DOWNLOAD_DIRECTORY)
        logger.info(f"VIIRS download directory: {cl.dir(VIIRS_download_directory)}")
        VIIRS_products_directory = join(sources_directory, DEFAULT_VIIRS_PRODUCTS_DIRECTORY)
        logger.info(f"VIIRS products directory: {cl.dir(VIIRS_products_directory)}")
        VIIRS_mosaic_directory = join(sources_directory, DEFAUL_VIIRS_MOSAIC_DIRECTORY)
        logger.info(f"VIIRS mosaic directory: {cl.dir(VIIRS_mosaic_directory)}")
        GEOS5FP_download_directory = join(sources_directory, DEFAULT_GEOS5FP_DOWNLOAD_DIRECTORY)
        logger.info(f"GEOS-5 FP download directory: {cl.dir(GEOS5FP_download_directory)}")
        GEOS5FP_products_directory = join(sources_directory, DEFAULT_GEOS5FP_PRODUCTS_DIRECTORY)
        logger.info(f"GEOS-5 FP products directory: {cl.dir(GEOS5FP_products_directory)}")
        VNP09GA_products_directory = join(sources_directory, DEFAULT_VNP09GA_PRODUCTS_DIRECTORY)
        logger.info(f"VNP09GA products directory: {cl.dir(VNP09GA_products_directory)}")
        VNP43NRT_products_directory = join(sources_directory, DEFAULT_VNP43NRT_PRODUCTS_DIRECTORY)
        logger.info(f"VNP43NRT products directory: {cl.dir(VNP43NRT_products_directory)}")

        # Re-check for existing product (double-check in case another process created it)
        if exists(L2T_STARS_zip_filename):
            logger.info(
                f"Found L2T STARS product zip: {cl.file(L2T_STARS_zip_filename)}"
            )
            return exit_code

        # Initialize HLS data connection
        logger.info(f"Connecting to CMR Search server: {CMR_SEARCH_URL}")
        try:
            HLS_connection = HLS2CMR(
                working_directory=working_directory,
                download_directory=HLS_download_directory,
                products_directory=HLS_products_directory,
                target_resolution=target_resolution,
            )
        except CMRServerUnreachable as e:
            logger.exception(e)
            raise AncillaryServerUnreachable(
                f"Unable to connect to CMR Search server: {CMR_SEARCH_URL}"
            )

        # Check if the tile is on land (HLS tiles cover land and ocean, STARS is for land)
        if not HLS_connection.tile_grid.land(tile=tile):
            raise LandFilter(f"Sentinel tile {tile} is not on land. Skipping processing.")

        # Initialize VIIRS data connections based on 'use_VNP43NRT' flag
        if use_VNP43NRT:
            try:
                NDVI_VIIRS_connection = VNP43NRT(
                    working_directory=working_directory,
                    download_directory=VIIRS_download_directory,
                    mosaic_directory=VIIRS_mosaic_directory,
                    GEOS5FP_download=GEOS5FP_download_directory,
                    GEOS5FP_products=GEOS5FP_products_directory,
                    VNP09GA_directory=VNP09GA_products_directory,
                    VNP43NRT_directory=VNP43NRT_products_directory,
                )

                albedo_VIIRS_connection = VNP43NRT(
                    working_directory=working_directory,
                    download_directory=VIIRS_download_directory,
                    mosaic_directory=VIIRS_mosaic_directory,
                    GEOS5FP_download=GEOS5FP_download_directory,
                    GEOS5FP_products=GEOS5FP_products_directory,
                    VNP09GA_directory=VNP09GA_products_directory,
                    VNP43NRT_directory=VNP43NRT_products_directory,
                )
            except CMRServerUnreachable as e:
                logger.exception(e)
                raise AncillaryServerUnreachable(f"Unable to connect to CMR search server for VNP43NRT.")
        else:
            try:
                NDVI_VIIRS_connection = VNP43IA4(
                    working_directory=working_directory,
                    download_directory=VIIRS_download_directory,
                    products_directory=VIIRS_products_directory,
                    mosaic_directory=VIIRS_mosaic_directory,
                )

                albedo_VIIRS_connection = VNP43MA3(
                    working_directory=working_directory,
                    download_directory=VIIRS_download_directory,
                    products_directory=VIIRS_products_directory,
                    mosaic_directory=VIIRS_mosaic_directory,
                )
            except LPDAACServerUnreachable as e:
                logger.exception(e)
                raise AncillaryServerUnreachable(f"Unable to connect to VIIRS LPDAAC server.")

        # Define date ranges for data retrieval and fusion
        end_date = date_UTC
        # The start date of the BRDF-corrected VIIRS coarse time-series is 'spinup_days' before the target date
        VIIRS_start_date = end_date - timedelta(days=spinup_days)
        # To produce that first BRDF-corrected image, VNP09GA (raw VIIRS) is needed starting 16 days prior to the first coarse date
        VIIRS_download_start_date = VIIRS_start_date - timedelta(days=16)
        VIIRS_end_date = end_date

        # Define start date of HLS fine image input time-series
        if using_prior and prior_date_UTC and prior_date_UTC >= VIIRS_start_date:
            # If a valid prior is used and its date is within or before the VIIRS start date,
            # HLS inputs begin the day after the prior
            HLS_start_date = prior_date_UTC + timedelta(days=1)
        else:
            # If no prior or prior is too old, HLS inputs begin on the same day as the VIIRS inputs
            HLS_start_date = VIIRS_start_date
        HLS_end_date = end_date # HLS end date is always the same as the target date

        logger.info(
            f"Processing STARS HLS-VIIRS NDVI and albedo for tile {cl.place(tile)} from "
            f"{cl.time(VIIRS_start_date)} to {cl.time(end_date)}"
        )

        # Get HLS listing to check for data availability
        try:
            HLS_listing = HLS_connection.listing(
                tile=tile, start_UTC=HLS_start_date, end_UTC=HLS_end_date
            )
        except HLSTileNotAvailable as e:
            logger.exception(e)
            raise LandFilter(f"Sentinel tile {tile} cannot be processed due to HLS tile unavailability.")
        except Exception as e:
            logger.exception(e)
            raise AncillaryServerUnreachable(
                f"Unable to scan Harmonized Landsat Sentinel server: {HLS_connection.remote}"
            )

        # Check for missing HLS Sentinel data
        missing_sentinel_dates = HLS_listing[HLS_listing.sentinel == "missing"].date_UTC
        if len(missing_sentinel_dates) > 0:
            raise AncillaryLatency(
                f"HLS Sentinel is not yet available at tile {tile} for dates: "
                f"{', '.join(missing_sentinel_dates.dt.strftime('%Y-%m-%d'))}"
            )

        # Log available HLS Sentinel data
        sentinel_listing = HLS_listing[~pd.isna(HLS_listing.sentinel)][
            ["date_UTC", "sentinel"]
        ]
        logger.info(f"HLS Sentinel is available on {cl.val(len(sentinel_listing))} dates:")
        for i, (list_date_utc, sentinel_granule) in sentinel_listing.iterrows():
            sentinel_filename = sentinel_granule["meta"]["native-id"]
            logger.info(f"* {cl.time(list_date_utc)}: {cl.file(sentinel_filename)}")

        # Check for missing HLS Landsat data
        missing_landsat_dates = HLS_listing[HLS_listing.landsat == "missing"].date_UTC
        if len(missing_landsat_dates) > 0:
            raise AncillaryLatency(
                f"HLS Landsat is not yet available at tile {tile} for dates: "
                f"{', '.join(missing_landsat_dates.dt.strftime('%Y-%m-%d'))}"
            )

        # Log available HLS Landsat data
        landsat_listing = HLS_listing[~pd.isna(HLS_listing.landsat)][
            ["date_UTC", "landsat"]
        ]
        logger.info(f"HLS Landsat is available on {cl.val(len(landsat_listing))} dates:")
        for i, (list_date_utc, landsat_granule) in landsat_listing.iterrows():
            landsat_filename = landsat_granule["meta"]["native-id"]
            logger.info(f"* {cl.time(list_date_utc)}: {cl.file(landsat_filename)}")

        # If only sources are requested, retrieve them and exit
        if sources_only:
            logger.info("Sources only flag enabled. Retrieving source data.")
            retrieve_STARS_sources(
                tile=tile,
                geometry=geometry,
                HLS_start_date=HLS_start_date,
                HLS_end_date=HLS_end_date,
                VIIRS_start_date=VIIRS_download_start_date,
                VIIRS_end_date=VIIRS_end_date,
                HLS_connection=HLS_connection,
                VIIRS_connection=NDVI_VIIRS_connection, # Use NDVI_VIIRS_connection as a general VIIRS connection
            )
            # Regenerate inputs to ensure all files are staged, even if not fused
            NDVI_coarse_geometry = HLS_connection.grid(tile=tile, cell_size=NDVI_resolution)
            albedo_coarse_geometry = HLS_connection.grid(tile=tile, cell_size=albedo_resolution)

            NDVI_coarse_directory = generate_NDVI_coarse_directory(
                input_staging_directory=input_staging_directory, tile=tile
            )
            NDVI_fine_directory = generate_NDVI_fine_directory(
                input_staging_directory=input_staging_directory, tile=tile
            )
            albedo_coarse_directory = generate_albedo_coarse_directory(
                input_staging_directory=input_staging_directory, tile=tile
            )
            albedo_fine_directory = generate_albedo_fine_directory(
                input_staging_directory=input_staging_directory, tile=tile
            )

            generate_STARS_inputs(
                tile=tile,
                date_UTC=date_UTC,
                HLS_start_date=HLS_start_date,
                HLS_end_date=HLS_end_date,
                VIIRS_start_date=VIIRS_start_date,
                VIIRS_end_date=VIIRS_end_date,
                NDVI_resolution=NDVI_resolution,
                albedo_resolution=albedo_resolution,
                target_resolution=target_resolution,
                NDVI_coarse_geometry=NDVI_coarse_geometry,
                albedo_coarse_geometry=albedo_coarse_geometry,
                working_directory=working_directory,
                NDVI_coarse_directory=NDVI_coarse_directory,
                NDVI_fine_directory=NDVI_fine_directory,
                albedo_coarse_directory=albedo_coarse_directory,
                albedo_fine_directory=albedo_fine_directory,
                HLS_connection=HLS_connection,
                NDVI_VIIRS_connection=NDVI_VIIRS_connection,
                albedo_VIIRS_connection=albedo_VIIRS_connection,
                calibrate_fine=calibrate_fine,
            )
        else:
            # Otherwise, proceed with full product processing
            process_STARS_product(
                tile=tile,
                date_UTC=date_UTC,
                time_UTC=time_UTC,
                build=build,
                product_counter=product_counter,
                HLS_start_date=HLS_start_date,
                HLS_end_date=HLS_end_date,
                VIIRS_start_date=VIIRS_start_date,
                VIIRS_end_date=VIIRS_end_date,
                NDVI_resolution=NDVI_resolution,
                albedo_resolution=albedo_resolution,
                target_resolution=target_resolution,
                working_directory=working_directory,
                model_directory=model_directory,
                input_staging_directory=input_staging_directory,
                L2T_STARS_granule_directory=L2T_STARS_granule_directory,
                L2T_STARS_zip_filename=L2T_STARS_zip_filename,
                L2T_STARS_browse_filename=L2T_STARS_browse_filename,
                metadata=metadata,
                prior=prior,
                HLS_connection=HLS_connection,
                NDVI_VIIRS_connection=NDVI_VIIRS_connection,
                albedo_VIIRS_connection=albedo_VIIRS_connection,
                using_prior=using_prior,
                calibrate_fine=calibrate_fine,
                remove_input_staging=remove_input_staging,
                remove_prior=remove_prior,
                remove_posterior=remove_posterior,
                threads=threads,
                num_workers=num_workers,
            )

    # --- Exception Handling for PGE ---
    except (ConnectionError, urllib.error.HTTPError, CMRServerUnreachable) as exception:
        logger.exception(exception)
        exit_code = ANCILLARY_SERVER_UNREACHABLE
    except DownloadFailed as exception:
        logger.exception(exception)
        exit_code = DOWNLOAD_FAILED
    except HLSBandNotAcquired as exception:
        logger.exception(exception)
        exit_code = DOWNLOAD_FAILED
    except HLSNotAvailable as exception:
        logger.exception(exception)
        exit_code = LAND_FILTER  # This might indicate no HLS data for the tile, similar to land filter
    except (HLSSentinelMissing, HLSLandsatMissing) as exception:
        logger.exception(exception)
        exit_code = ANCILLARY_LATENCY
    except ECOSTRESSExitCodeException as exception:
        # Catch custom ECOSTRESS exceptions and use their defined exit code
        logger.exception(exception)
        exit_code = exception.exit_code
    except Exception as exception:
        # Catch any other unexpected exceptions
        logger.exception(exception)
        exit_code = UNCLASSIFIED_FAILURE_EXIT_CODE

    logger.info(f"L2T_STARS exit code: {exit_code}")
    return exit_code


def main():
    """
    Main function for parsing command-line arguments and running the L2T_STARS PGE.

    This function uses `argparse` for robust command-line argument parsing,
    providing a clear interface for users to specify the run-configuration file
    and other optional parameters.
    """
    parser = argparse.ArgumentParser(
        description="ECOSTRESS Collection 3 L2T_STARS PGE for generating Tiled Ancillary NDVI and Albedo products.",
        formatter_class=argparse.RawTextHelpFormatter, # Allows for more flexible help text formatting
        epilog=f"L2T_STARS PGE Version: {PGEVersion}\n\n"
               "Example usage:\n"
               "  python {sys.argv[0]} --runconfig /path/to/RunConfig.xml\n"
               "  python {sys.argv[0]} --runconfig /path/to/RunConfig.xml --date 2023-01-15\n"
               "  python {sys.argv[0]} --runconfig /path/to/RunConfig.xml --sources-only\n"
    )

    # Positional argument for the runconfig file
    parser.add_argument(
        "runconfig",
        type=str,
        help="Path to the XML run-configuration file.",
    )

    # Optional arguments
    parser.add_argument(
        "--date",
        type=str,
        help="Target UTC date for product generation (YYYY-MM-DD). Overrides date in runconfig.",
        metavar="YYYY-MM-DD"
    )
    parser.add_argument(
        "--spinup-days",
        type=int,
        default=DEFAULT_SPINUP_DAYS,
        help=f"Number of days for the VIIRS time-series spin-up. Defaults to {DEFAULT_SPINUP_DAYS} days.",
        metavar="DAYS"
    )
    parser.add_argument(
        "--target-resolution",
        type=int,
        default=DEFAULT_TARGET_RESOLUTION,
        help=f"Desired output product resolution in meters. Defaults to {DEFAULT_TARGET_RESOLUTION}m.",
        metavar="METERS"
    )
    parser.add_argument(
        "--ndvi-resolution",
        type=int,
        default=DEFAULT_NDVI_RESOLUTION,
        help=f"Resolution of coarse NDVI data in meters. Defaults to {DEFAULT_NDVI_RESOLUTION}m.",
        metavar="METERS"
    )
    parser.add_argument(
        "--albedo-resolution",
        type=int,
        default=DEFAULT_ALBEDO_RESOLUTION,
        help=f"Resolution of coarse albedo data in meters. Defaults to {DEFAULT_ALBEDO_RESOLUTION}m.",
        metavar="METERS"
    )
    parser.add_argument(
        "--use-vnp43nrt",
        action="store_true",
        default=DEFAULT_USE_VNP43NRT,
        help=f"Use VNP43NRT for VIIRS products. Defaults to {'True' if DEFAULT_USE_VNP43NRT else 'False'}.",
    )
    parser.add_argument(
        "--no-vnp43nrt",
        action="store_false",
        dest="use_vnp43nrt", # This argument sets use_vnp43nrt to False
        help="Do NOT use VNP43NRT for VIIRS products. Use VNP43IA4/VNP43MA3 instead.",
    )
    parser.add_argument(
        "--calibrate-fine",
        action="store_true",
        default=DEFAULT_CALIBRATE_FINE,
        help=f"Calibrate fine resolution HLS data to coarse resolution VIIRS data. Defaults to {'True' if DEFAULT_CALIBRATE_FINE else 'False'}.",
    )
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Only retrieve and stage source data (HLS, VIIRS); do not perform data fusion or generate final product.",
    )
    parser.add_argument(
        "--no-remove-input-staging",
        action="store_false",
        dest="remove_input_staging",
        default=True,
        help="Do NOT remove the input staging directory after processing.",
    )
    parser.add_argument(
        "--no-remove-prior",
        action="store_false",
        dest="remove_prior",
        default=True,
        help="Do NOT remove prior intermediate files after use.",
    )
    parser.add_argument(
        "--no-remove-posterior",
        action="store_false",
        dest="remove_posterior",
        default=True,
        help="Do NOT remove posterior intermediate files after product generation.",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="auto",
        help='Number of Julia threads to use, or "auto". Defaults to "auto".',
        metavar="COUNT"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=f"Number of Julia workers for distributed processing. Defaults to 4.",
        metavar="COUNT"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PGEVersion}",
        help="Show program's version number and exit.",
    )

    args = parser.parse_args()

    # Call the main L2T_STARS processing function with parsed arguments
    exit_code = L2T_STARS(
        runconfig_filename=args.runconfig,
        date_UTC=args.date,
        spinup_days=args.spinup_days,
        target_resolution=args.target_resolution,
        NDVI_resolution=args.ndvi_resolution,
        albedo_resolution=args.albedo_resolution,
        use_VNP43NRT=args.use_vnp43nrt,
        calibrate_fine=args.calibrate_fine,
        sources_only=args.sources_only,
        remove_input_staging=args.remove_input_staging,
        remove_prior=args.remove_prior,
        remove_posterior=args.remove_posterior,
        threads=args.threads,
        num_workers=args.num_workers,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()