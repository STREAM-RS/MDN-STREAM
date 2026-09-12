import warnings
import os

# Enforce legacy Keras 2 engine for H5 model deserialization
os.environ["TF_USE_LEGACY_KERAS"] = "1"

warnings.simplefilter(action='ignore', category=FutureWarning)

from .__version__ import __version__
from .product_estimation import image_estimates, get_estimates
from .meta import get_sensor_bands
from .utils import get_tile_data, current_support, download_example_imagery, mask_land, get_tile_geographic_info, print_available_imagery,IMAGERY_REGISTRY
from .tile_utils import load_lonlat, extract_satellite_data, translate_wavelengths_to_landsat_bands, export_dataset,fix_geo_orientation
from .utils import write_cube_to_netcdf4, generate_config
from .gloria_processing_utils import get_gloria_trainTestData, resample_Rrs
from .parameters import get_args
from .utilities import get_mdn_preds, get_mdn_preds_raw, get_mdn_preds_uncertainties, map_cube_mdn_full                  #, get_mdn_uncert_ensemble, get_mdn_preds_uncertainties, map_cube_mdn_full
from .plot_utilities import create_scatterplots_trueVsPred, display_sat_rgb, find_rgb_img, \
    overlay_rgb_mdnProducts, create_scatterplots_axis, create_performance_plots, overlay_rgb_mdn_preds_limits
from .metrics import performance, mdsa, sspb, slope, rmsle
from .benchmarks.chl.OC.model import OC as OC
from .user_utilities import get_spectral_preds, get_spectral_preds_raw,  map_cube_mdn, load_water_quality_csv
