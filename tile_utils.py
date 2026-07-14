import numpy as np
from pathlib import Path
import rasterio
import xarray as xr
import rioxarray
from functools import lru_cache
from netCDF4 import Dataset
from pyproj import Proj
import dask.array as da
from rasterio.transform import xy
from rasterio.enums import Resampling
import dask

import sys
from typing import Union, Optional, List
import json
import re
from pathlib import Path

from .meta import get_sensor_bands, check_sensor_availability

import warnings
warnings.filterwarnings('ignore')


def translate_wavelengths_to_landsat_bands(requested_bands, wavelength_unit="nm", tolerance=20):
    """
    Translates input center wavelengths into official Landsat 8/9 OLI string band names.
    
    Parameters
    ----------
    requested_bands : list of float
        The list of target center wavelengths to resolve.
    wavelength_unit : str, optional
        'nm' for nanometers or 'um' for microns. Default is 'nm'.
    tolerance : float, optional
        Maximum window allowed to map a wavelength to a sensor band center.
        
    Returns
    -------
    landsat_bands : list of str
        List of matching string band identifiers (e.g., ['B01', 'B02', 'B04'])
    """
    # Official Landsat 8/9 OLI operational band center wavelengths
    landsat_oli_lookup = {
        "B1": {"name": "Coastal Aerosol", "center_nm": 443, "center_um": 0.443},
        "B2": {"name": "Blue",            "center_nm": 482, "center_um": 0.482},
        "B3": {"name": "Green",           "center_nm": 561, "center_um": 0.561},
        "B4": {"name": "Red",             "center_nm": 655, "center_um": 0.655},
        "B5": {"name": "NIR",             "center_nm": 865, "center_um": 0.865},
        "B6": {"name": "SWIR 1",          "center_nm": 1609, "center_um": 1.609},
        "B7": {"name": "SWIR 2",          "center_nm": 2201, "center_um": 2.201},
        "B9": {"name": "Cirrus",          "center_nm": 1373, "center_um": 1.373},
    }
    
    unit_key = "center_nm" if wavelength_unit.lower() in ["nm", "nanometer"] else "center_um"
    landsat_bands = []
    
    for wl in requested_bands:
        # Evaluate spatial distance to every valid OLI band channel
        diffs = {b_code: np.abs(meta[unit_key] - wl) for b_code, meta in landsat_oli_lookup.items()}
        best_match_code = min(diffs, key=diffs.get)
        
        if diffs[best_match_code] > tolerance:
            raise ValueError(
                f"Wavelength {wl} {wavelength_unit} does not fall within the "
                f"±{tolerance} window of any Landsat band center."
            )
            
        landsat_bands.append(best_match_code)
        
    return landsat_bands


class DatasetBridge:

    def __init__(self, filename, *args, **kwargs):
        self.filename = Path(filename)

    def __enter__(self):
        if self.filename.suffix == '.nc':
            self.data = Dataset(self.filename.as_posix(), 'r')
        elif self.filename.suffix in ['.hdr', '.bsq']:
            self.data = rasterio.open(self.filename.with_suffix('.bsq'))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data.close()

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    @property
    def groups(self):
        if hasattr(self.data, 'groups'):
            return self.data.groups
        return {}

    @property
    def variables(self):
        if hasattr(self.data, 'variables'):
            return self.data.variables
        return {str(k): None for k in ['lon', 'lat'] + self.bands}

    @property
    @lru_cache()
    def bands(self):
        assert (hasattr(self.data, 'tags')), f'Cannot get bands from {self.filename}: band tags unavailable'
        # extract = lambda s: float( re.findall(r'^.*\((\d*\.\d+)\)$', s)[0] ) # e.g. 'Band Math (band 235 refl [%*100]:L1C_2019-08-04_Trasimeno_fsr_CAL_atm.bsq) (0.999500)'
        # return [int(extract(desc) * 1000) for desc in self.data.descriptions]
        extract = lambda i: int(float(self.data.tags(i)['wavelength']) * 1000)
        return [f'Rrs_{extract(i + 1)}' for i in range(self.data.count)]

    @property
    @lru_cache()
    def lonlat(self):
        assert (hasattr(self.data, 'crs')), f'Cannot get latlon from {self.filename}: CRS unavailable'
        proj = Proj(self.data.crs)
        grid = self.data.transform * np.meshgrid(
            *map(lambda x: np.arange(x) + 0.5, [self.data.width, self.data.height]))
        return dict(zip(['lon', 'lat'], grid))
        return dict(zip(['lon', 'lat'], proj(*grid, inverse=True)))

    def __getitem__(self, key):
        if hasattr(self.data, 'variables'):
            return self.data[key]

        if key in ['lon', 'lat']:
            item = self.lonlat[key]
        else:
            item = self.data.read(self.bands.index(key) + 1)

        item = np.ma.array(item)
        item.set_auto_mask = lambda *args, **kwargs: None
        item.set_always_mask = lambda *args, **kwargs: None
        return item


def load_lonlat(file_name, center=False):
    """
    This function can be used extract the longitude and latitude information from the NetCDF files

    :param file_name (str, pathlib.Path)
    The file address of the netCDF4 file

    :param center (bool) [Default: False]
    A boolean Flag which decides if the image needs to be centered at a specific location

    :return lon (np.ndarray)
    A 2D matrix which holds longitude information for each pixel

    :return lat (np.ndarray)
    A 2D matrix which holds latitude information for each pixel

    :return extent (group)group variable with the NSEW extents of the image
    """

    with DatasetBridge(file_name) as data:
        if 'navigation_data' in data.groups.keys():
            data = data['navigation_data']
        lon_k, lat_k = ('lon', 'lat') if 'lon' in data.variables.keys() else ('longitude', 'latitude')
        lon,   lat   = data[lon_k][:], data[lat_k][:]

    if len(lon.shape) == 1:
        lon, lat = np.meshgrid(lon, lat)

    lon[lon < -180] = np.nan
    lat[lat < -180] = np.nan
    lon.mask = False
    lat.mask = False

    if center:
        lon_orig, lat_orig   = lon, lat
        img_param = center_image(lon, lat)
        lon, lat = lon[img_param], lat[img_param]

    # left, right, bottom, top | west, east, south, north
    extent = np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)
    return lon, lat, extent


def center_image(im_lon, im_lat, location=None, img_width=1500, img_height=2500):
    """
    Extracted from Brandon Smith's Plot toolbox

    """
    center_lon = im_lon[im_lon.shape[0] // 2, im_lon.shape[1] // 2]
    center_lat = im_lat[im_lat.shape[0] // 2, im_lat.shape[1] // 2]

    partialy = np.argmin(np.abs(im_lat[:, im_lat.shape[1] // 2] - center_lat))
    partialx = np.argmin(np.abs(im_lon[partialy] - center_lon))
    partialy = np.argmin(np.abs(im_lat[:, partialx] - center_lat))
    partialx = slice(max(0, partialx - img_width), min(im_lon.shape[1], partialx + img_width))
    partialy = slice(max(0, partialy - img_height), min(im_lat.shape[0], partialy + img_height))

    if (partialx.stop - partialx.start) <= 0 or (partialy.stop - partialy.start) <= 0:
        print('Slices (y/lat, x/lon):', partialy, partialx)
        print(f'Lat shape={im_lat.shape} [min, max]=[{im_lat.min()}, {im_lat.max()}]')
        print(f'Lon shape={im_lon.shape} [min, max]=[{im_lon.min()}, {im_lon.max()}]')
        coord_def = f'Center=({center_lat}, {center_lon})'
        raise Exception(f'No pixels within bounds!!')

    return partialy, partialx


def subset_bbox_2d(
    da: xr.DataArray,
    bbox: tuple[float, float, float, float],
) -> xr.DataArray:
    """
    Subset a swath-based DataArray using a geographic bounding box,
    trimming excess rows and columns introduced by 2D masking.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with 2D latitude/longitude coordinates.
    bbox : tuple
        Bounding box as (lat_min, lat_max, lon_min, lon_max).

    Returns
    -------
    xr.DataArray
        Tightly cropped DataArray within the bounding box.
    """

    lat_min, lat_max, lon_min, lon_max = bbox

    # Ensure lat/lon are coordinates
    if "latitude" not in da.coords or "longitude" not in da.coords:
        raise ValueError("latitude and longitude must be coordinates of da")

    #Generate the geographic mask
    mask = (
            (da.latitude >= lat_min) & (da.latitude <= lat_max) &
            (da.longitude >= lon_min) & (da.longitude <= lon_max)
    )

    da_masked = da.where(mask)

    # Find valid rows/cols
    valid_y = da_masked.notnull().any(dim=[d for d in da.dims if d != "y"])
    valid_x = da_masked.notnull().any(dim=[d for d in da.dims if d != "x"])

    # Index-based trimming
    y_idx = np.where(valid_y)[0]
    x_idx = np.where(valid_x)[0]

    if len(y_idx) == 0 or len(x_idx) == 0:
        raise ValueError("Bounding box does not intersect the data.")

    return da_masked.isel(
        y=slice(y_idx.min(), y_idx.max() + 1),
        x=slice(x_idx.min(), x_idx.max() + 1),
    )


def extract_netcdf_satellite_rrs(
    nc_file: Union[str, Path],
    sensor: str,
    bands=None,
    wavelength_unit="nm",
    flag_toa: bool = False,
    verbose: bool = False
) -> xr.DataArray:
    """
    Extract satellite-derived reflectance from a SeaDAS-style NetCDF Level-2 product 
    by automatically detecting whether Bottom-of-Atmosphere (Rrs/Rw) or 
    Top-of-Atmosphere (rhot/rho_t) variables are present.

    The function dynamically adapts to both the hierarchical structure (nested groups 
    vs. flat root level) and the variable naming convention of the file. If 'Rw' 
    variables are detected, they are automatically scaled to standard Rrs equivalents 
    using the transformation: Rrs = Rw / pi.

    Parameters
    ----------
    nc_file : str or pathlib.Path
        Path to the SeaDAS NetCDF file.
    sensor : str
        Sensor identifier used to determine the expected wavelength bands.
    bands : list of float or None, optional
        Target center wavelengths to load (expressed in `wavelength_unit`). 
        If None, all available image bands are retrieved. Default is None.
    wavelength_unit : str, optional
        Unit format for center wavelengths. Supported options are 'nm' 
        (nanometers) or 'um' (microns). Default is "nm".
    flag_toa : bool, optional
        If True, the function switches focus to extract Top-of-Atmosphere (TOA) 
        radiance ('rhot', 'Rhot', 'rho_t', 'RHOT') instead of 
        standard reflectance Rrs products. Default is False.
    verbose : bool, optional
        If True, prints diagnostic information about the detected file format, 
        variable prefixes, and extracted dimensions. Default is False.

    Returns
    -------
    xr.DataArray
        Refance data array with dimensions (band, y, x) and associated 2D 
        spatial latitude/longitude coordinates mapping along with written CRS metadata.

    Raises
    ------
    FileNotFoundError
        Raised if the target NetCDF file does not exist on the file system.
    ValueError
        Raised if requested variables are not identified within the target file, 
        or if expected wavelength bands cannot be matched.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    nc_file = Path(nc_file)
    if not nc_file.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {nc_file}")

    if bands is None:
        try:
            bands = get_sensor_bands(sensor)
        except NameError:
            pass

    # ------------------------------------------------------------------
    # Read NetCDF contents lazily using xarray + dask backends
    # ------------------------------------------------------------------
    ds = xr.open_dataset(nc_file, chunks={})
    ac_str = str(ds.attrs.get("generated_by", ds.attrs.get("processor", "unknown_source")))
    
    try:
        geo_source = xr.open_dataset(nc_file, group="geophysical_data", chunks={})
        nav_source = xr.open_dataset(nc_file, group="navigation_data", chunks={})
        if verbose:
            print("Hierarchical group schema detected. Accessing nested groups.")
    except Exception:
        if verbose:
            print("Flat group schema detected. Accessing root level.")
        geo_source = ds
        nav_source = ds

    try:
        wav_source = xr.open_dataset(nc_file, group="sensor_band_parameters", chunks={})
    except Exception:
        wav_source = None

    # --------------------------------------------------------------
    # Navigation / Coordinate Extraction (Lazy metadata parsing)
    # --------------------------------------------------------------
    lat_keys = ["latitude", "lat", "LATITUDE", "LAT"]
    lon_keys = ["longitude", "lon", "LONGITUDE", "LON"]

    lat_key = next((k for k in lat_keys if k in nav_source.variables), None)
    lon_key = next((k for k in lon_keys if k in nav_source.variables), None)

    if not lat_key or not lon_key:
        raise KeyError(f"Could not find valid spatial coordinate variables. Available keys: {list(nav_source.variables.keys())}")

    lat = nav_source[lat_key]
    lon = nav_source[lon_key]

    if lat.ndim == 1 and lon.ndim == 1:
        if verbose:
            print("1D Coordinate arrays detected. Broadcasting to 2D grid meshes.")
        lon, lat = xr.broadcast(lon, lat)

    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))

    # --------------------------------------------------------------
    # CRS Extraction
    # --------------------------------------------------------------
    crs_detected = None
    
    # 1. Try to read CRS from the dataset via rioxarray
    if hasattr(ds, "rio") and ds.rio.crs is not None:
        crs_detected = ds.rio.crs
    elif hasattr(geo_source, "rio") and geo_source.rio.crs is not None:
        crs_detected = geo_source.rio.crs
        
    # 2. Fallback check on attributes or assign standard WGS84 for SeaDAS L2
    if crs_detected is None:
        # Check standard CF metadata attributes
        crs_wkt = ds.attrs.get("crs_wkt", geo_source.attrs.get("grid_mapping", None))
        if crs_wkt:
            crs_detected = crs_wkt
        else:
            # SeaDAS Level-2 products are default unprojected WGS 84
            crs_detected = "EPSG:4326"
            
    if verbose:
        print(f"Extracted/Assigned Dataset CRS: {crs_detected}")

    # --------------------------------------------------------------
    # Auto-detect Variable Schema (Multi-band Array vs. Per-Band Vars)
    # --------------------------------------------------------------
    if flag_toa:
        multi_band_keys = ["rhot", "Rhot", "rho_t", "RHOT", "rhot_bands"]
        default_var_name = "rhot"
        long_name_default = "Top-of-atmosphere radiance"
    else:
        multi_band_keys = ["Rrs", "Rw", "Rrs_bands", "reflectance"]
        default_var_name = "Rrs"
        long_name_default = "Remote-sensing reflectance"

    target_multi_key = next((k for k in multi_band_keys if k in geo_source.variables), None)
    
    # Scenario A: Integrated Multi-Band 3D Array found
    if target_multi_key and geo_source[target_multi_key].ndim >= 3:
        if flag_toa:
            variable_prefix = "rhot"
        else:
            variable_prefix = "Rw" if "rw" in target_multi_key.lower() else "Rrs"
            
        if verbose:
            print(f"Detected multi-band array variable: '{target_multi_key}' (Prefix: {variable_prefix})")
        
        raw_shape = geo_source[target_multi_key].shape
        wave_keys = ["wavelength_3d", "wavelength", "wavelengths", "band", "bands", "wvl"]
        wave_key = None
        wave_origin = None

        for src in [wav_source, geo_source, ds]:
            if src is not None:
                for k in wave_keys:
                    if k in src.variables:
                        w_size = src[k].shape[0]
                        if w_size in raw_shape:
                            wave_key = k
                            wave_origin = src
                            break
                if wave_key:
                    break
        
        if not wave_key:
            for src in [wav_source, geo_source, ds]:
                if src is not None:
                    wave_key = next((k for k in wave_keys if k in src.variables), None)
                    if wave_key:
                        wave_origin = src
                        break

        if wave_key and wave_origin:
            wavelengths = wave_origin[wave_key].values
            if verbose:
                print(f"Selected wavelength coordinate key: '{wave_key}' matching cube dimensions.")
        else:
            raise ValueError(f"Found multi-band variable '{target_multi_key}' with shape {raw_shape}, "
                             f"but could not locate a matching wavelength dimension array.")

        raw_data = geo_source[target_multi_key]
        
        if raw_data.shape[-1] == len(wavelengths):
            dims_list = list(raw_data.dims)
            band_dim = dims_list[-1]
            other_dims = dims_list[:-1]
            raw_data = raw_data.transpose(band_dim, *other_dims)
            
        if bands is None:
            bands = wavelengths.tolist()
            reflectance_data = raw_data
        else:
            selected_indices = []
            for b in bands:
                idx = np.argmin(np.abs(wavelengths - b))
                if np.abs(wavelengths[idx] - b) >= 5:
                    raise ValueError(f"No band found matching target {b}nm. Available: {wavelengths}")
                selected_indices.append(idx)
            reflectance_data = raw_data[selected_indices, ...]
            
        selected_vars = [f"{target_multi_key}[{b}]" for b in bands]

    # Scenario B: Individual Per-Band Variables
    else:
        if flag_toa:
            pattern = re.compile(r'^(rhot|rho_t)_?(\d+)$', re.IGNORECASE)
            valid_prefixes = ["RHOT", "RHO_T"]
        else:
            pattern = re.compile(r'^(rrs|rw)_?(\d+)$', re.IGNORECASE)
            valid_prefixes = ["RRS", "RW"]
        
        matched_vars = []
        for name in geo_source.variables:
            match = pattern.match(name)
            if match:
                prefix = match.group(1).upper()
                wave = int(match.group(2))
                matched_vars.append((name, prefix, wave))

        # Determine dominant variable type present
        found_prefix = next((p for p in valid_prefixes if any(m[1] == p for m in matched_vars)), None)

        if found_prefix:
            dominant_filter = found_prefix
            variable_prefix = found_prefix.lower() if flag_toa else ("Rw" if found_prefix == "RW" else "Rrs")
            if verbose:
                print(f"Auto-detected variable type: {dominant_filter}")
        else:
            target_desc = "TOA ('rhot_') " if flag_toa else "surface ('rrs_' or 'rw_')"
            raise ValueError(
                f"Invalid NetCDF format: Could not find any target {target_desc} variables "
                f"in the dataset. Available variables: {list(geo_source.variables.keys())}"
            )

        filtered_vars = sorted(
            [m for m in matched_vars if m[1] == dominant_filter], 
            key=lambda x: x[2]
        )
        
        reflectance_names = [m[0] for m in filtered_vars]
        wavelengths = np.array([m[2] for m in filtered_vars])

        if bands is None:
            bands = wavelengths.tolist()

        selected_vars = []
        for band in bands:
            idx = np.argmin(np.abs(wavelengths - band))
            if np.abs(wavelengths[idx] - band) >= 5:
                raise ValueError(f"No band found corresponding to {variable_prefix}_{band}. Available bands: {wavelengths}")
            selected_vars.append(reflectance_names[idx])

        reflectance_data = xr.concat(
            [geo_source[var] for var in selected_vars], 
            dim="band"
        )

    # Apply oceanographic conversion rule lazily if raw format is surface Rw
    if not flag_toa and variable_prefix == "Rw":
        if verbose:
            print("Converting Water-Leaving Reflectance (Rw) to Remote-Sensing Reflectance (Rrs) via / pi rule...")
        reflectance_data = reflectance_data / np.pi
        long_name_default = "Remote-sensing reflectance (Pi-corrected from rw)"

    if verbose:
        print(f"Selected variables: {selected_vars}")
        print(f"Output spectral shape: {reflectance_data.shape} (band, y, x)")
        print(f"Latitude/Longitude shape: {lat.shape}")

    # Build the final DataArray
    da_final = xr.DataArray(
        reflectance_data.data,
        dims=("band", "y", "x"),
        coords={
            "band": bands,
            "latitude": (("y", "x"), lat.data),
            "longitude": (("y", "x"), lon.data),
        },
        name=default_var_name,
        attrs={
            "sensor": sensor,
            "long_name": long_name_default,
            "units": "sr-1" if not flag_toa else "dimensionless",
            "source": f"SeaDAS Level-2 (Auto-detected {variable_prefix} source)",
            "ac_processor": ac_str,
            "extent": extent,
            "crs": str(crs_detected) # Save string representation to basic attributes
        },
    )

    # Use rioxarray to systematically register the CRS inside the array spatial properties
    da_final = da_final.rio.write_crs(crs_detected)

    return da_final.chunk({"band": -1, "y": 100, "x": 100})  # Chunking for dask parallelism


def extract_composite_geotiff_wjson(
    file_path,
    json_metadata_path,
    sensor: str,
    bands=None,
    wavelength_unit="nm",
    tolerance=None,
    rhow_flag: bool = False,
    flag_toa: bool = False,
) -> xr.DataArray:
    """
    Load a Planet SuperDove type GeoTIFF using rioxarray, applying metadata 
    scaling, band selection, coordinate grid construction, and CRS registration.

    Parameters
    ----------
    file_path : str or Path
        Path to the multi-band GeoTIFF reflectance file.
    json_metadata_path : str or Path
        Path to the STAC JSON metadata sidecar file matching the image asset.
    sensor : str
        Sensor identifier indicating the deployment origin platform.
    bands : list of float or None, optional
        Target center wavelengths to load (expressed in `wavelength_unit`). 
    wavelength_unit : str, optional
        Unit format for center wavelengths. Supported options are 'nm' or 'um'.
    tolerance : float, optional
        Acceptable search window radius to associate raster bands.
    rhow_flag : bool, optional
        If True, and flag_toa is False, incoming data arrays are divided by Pi 
        to convert hemispherical surface reflectance into standard Rrs. Default is False.
    flag_toa : bool, optional
        If True, notes the asset type as Top-of-Atmosphere (TOA) reflectance,
        updating dataset attributes and overriding surface-level Rrs scaling constraints.
        Default is False.

    Returns
    -------
    da : xarray.DataArray
        Reflectance data array with written CRS metadata and projection properties.
    """
    
    # 1. Load JSON metadata
    with open(json_metadata_path) as f:
        meta = json.load(f)

    asset_key = f"{Path(file_path).stem}_tif"
    if asset_key not in meta["assets"]:
        asset_key = list(meta["assets"].keys())[0]
        
    asset = meta["assets"][asset_key]
    
    # Validate Asset Role against TOA processing directives
    asset_roles = asset.get("roles", [])
    if flag_toa and any("reflectance" in str(role).lower() for role in asset_roles):
        raise ValueError(
            f"Conflict detected: 'flag_toa' is configured as True, but the target"
            f"metadata asset roles describe the product type as {asset_roles}."
        )

    scales = [b["scale"] for b in asset["raster:bands"]]
    center_wavelengths = [b["center_wavelength"] for b in asset["eo:bands"]]

    if wavelength_unit.lower() in ["nm", "nanometer", "nanometers"]:
        center_wavelengths = [w * 1000 for w in center_wavelengths]
        if tolerance is None:
            tolerance = 3
    else:
        if tolerance is None:
            tolerance = 0.003

    # 2. Open the GeoTIFF lazily using rioxarray + dask chunks
    da_img = xr.open_dataarray(file_path, chunks={}, mask_and_scale=True)

    # --------------------------------------------------------------
    # CRS Extraction
    # --------------------------------------------------------------
    # Extract CRS from the source dataset loaded via rioxarray
    crs_detected = None
    if hasattr(da_img, "rio") and da_img.rio.crs is not None:
        crs_detected = da_img.rio.crs
    else:
        # Check STAC JSON metadata sidecar parameters as a fallback (often EPSG integer)
        proj_epsg = meta.get("properties", {}).get("proj:epsg", None)
        if proj_epsg:
            crs_detected = f"EPSG:{proj_epsg}"
        else:
            # Fallback to standard web-mercator/UTM check or WGS84
            crs_detected = "EPSG:4326"

    # 3. Determine band indexes and map wavelengths
    if bands is None:
        selected_bands = center_wavelengths
        selected_scales = scales
        band_indices = list(range(1, len(center_wavelengths) + 1))
    else:
        band_indices, selected_scales, selected_bands = [], [], []
        for br in bands:
            diffs = np.abs(np.array(center_wavelengths) - br)
            idx = np.argmin(diffs)
            if diffs[idx] > tolerance:
                raise ValueError(
                    f"No band found within ±{tolerance} {wavelength_unit} of requested {br}. "
                    f"Available assets: {center_wavelengths}"
                )
            band_indices.append(idx + 1)
            selected_scales.append(scales[idx])
            selected_bands.append(center_wavelengths[idx])

    # Slice down to requested bands safely
    da_img = da_img.sel(band=band_indices)

    # 4. Apply scale factors using a raw NumPy array along the band axis
    scale_vector = np.array(selected_scales)[:, np.newaxis, np.newaxis] # Shape (B, 1, 1)
    
    if rhow_flag and not flag_toa:
        scaled_data = (da_img.data * scale_vector) / np.pi
        long_name = "Remote-sensing reflectance (Pi-corrected from rhow)"
    else:
        scaled_data = da_img.data * scale_vector
        long_name = "Top-of-atmosphere reflectance" if flag_toa else "Remote-sensing reflectance"

    var_name = "rhot" if flag_toa else "Rrs"
    units = "dimensionless" if flag_toa else "sr-1"
    source_str = f"STAC Asset Collection {'TOA' if flag_toa else 'BOA'} via rioxarray ({wavelength_unit})"

    # 5. Broadcast 2D spatial coordinate meshes explicitly from the clean x/y axes
    lon_grid, lat_grid = xr.broadcast(da_img.y, da_img.x)

    # Construct clean DataArray container with everything wired to Dask arrays
    da_final = xr.DataArray(
        scaled_data,
        dims=("band", "y", "x"),
        coords={
            "band": selected_bands,
            "latitude": (("y", "x"), lat_grid.data),
            "longitude": (("y", "x"), lon_grid.data),
            # Maintain spatial x and y coordinates for rioxarray compatibility
            "y": da_img.y,
            "x": da_img.x,
        },
        name=var_name,
        attrs={
            "sensor": sensor,
            "long_name": long_name,
            "units": units,
            "source": source_str,
            "extent": da_img.rio.bounds(),
            "crs": str(crs_detected) # Explicit string representation in attributes
        },
    )

    # Systematically write the CRS structure using rioxarray
    da_final = da_final.rio.write_crs(crs_detected)

    return da_final.chunk({"band": -1, "y": 100, "x": 100})  # Chunking for dask parallelism


def extract_band_geotiff(
    file_path,
    bands,
    sensor: str,
    wavelength_unit="nm",
    tolerance=None,
    rhow_flag: bool = False,
    flag_toa: bool = False,
) -> xr.DataArray:
    """
    Load from a folder of individual single-band GeoTIFFs matching a specified list of 
    key identifiers (e.g., wavelengths like '443' or band names like 'B01') using rioxarray.

    Parameters
    ----------
    file_path : str or Path
        Path to the directory containing the single-band GeoTIFF files.
    bands : list of str or list of int/float
        List of key strings or identifiers used to match and extract target files. 
    sensor : str
        Sensor identifier indicating the deployment origin platform.
    wavelength_unit : str, optional
        Preserved parameter for pipeline uniformity. Default is "nm".
    tolerance : float, optional
        Preserved parameter for pipeline uniformity. Default is None.
    rhow_flag : bool, optional
        If True, and flag_toa is False, incoming data arrays are divided by Pi 
        to convert hemispherical surface reflectance into standard Rrs. Default is False.
    flag_toa : bool, optional
        If True, tracks directory items as Top-of-Atmosphere (TOA) radiance/reflectance 
        products, assigning metadata structures accordingly. Default is False.

    Returns
    -------
    da : xarray.DataArray
        Reflectance data array with written CRS metadata and projection properties.
    """
    folder_path = Path(file_path)
    band_dataarrays = []
    final_coords_keys = []
    crs_detected = None

    band_names = [str(b) if not isinstance(b, str) else b for b in bands]

    for key_str in band_names:
        patterns = [
            f"*{key_str}.[tT][iI][fF]",
            f"*{key_str}.[tT][iI][fF][fF]",
            f"*{key_str}nm.[tT][iI][fF]",
            f"*{key_str}nm.[tT][iI][fF][fF]",
            f"*_{key_str}.[tT][iI][fF]",
            f"*_{key_str}.[tT][iI][fF][fF]"
        ]
        
        matched_files = []
        for pattern in patterns:
            matched_files.extend(list(folder_path.glob(pattern)))
            
        matched_files = list(set(matched_files))

        if not matched_files:
            raise FileNotFoundError(
                f"Could not locate a matching GeoTIFF for identifier key '{key_str}' in directory: {file_path}"
            )
            
        band_path = matched_files[0]

        da_band = xr.open_dataarray(band_path, chunks={}, mask_and_scale=True)
        da_band = da_band.where(da_band != -32767.0)
        
        # Capture the CRS from the first band we successfully read
        if crs_detected is None and hasattr(da_band, "rio") and da_band.rio.crs is not None:
            crs_detected = da_band.rio.crs
        
        if "band" in da_band.coords:
            da_band = da_band.squeeze("band", drop=True)
            
        band_dataarrays.append(da_band)
        del da_band
        
        try:
            if "." in key_str:
                final_coords_keys.append(float(key_str))
            else:
                final_coords_keys.append(int(key_str))
        except ValueError:
            final_coords_keys.append(key_str)

    # Fallback to standard WGS84 if no CRS could be extracted
    if crs_detected is None:
        crs_detected = "EPSG:4326"

    # Concatenate along the band dimension now that shapes are guaranteed to match
    da_img = xr.concat(band_dataarrays, dim="band")
    da_img = da_img.assign_coords(band=final_coords_keys)

    # Dynamic naming and metadata based on TOA vs Surface configurations
    if flag_toa:
        var_name = "rhot"
        long_name = "Top-of-atmosphere reflectance"
        units = "dimensionless"
        source_str = f"Custom Key-Matched TOA Raster Stack Collection via rioxarray"
    else:
        if rhow_flag:
            da_img = da_img / np.pi
            long_name = "Remote-sensing reflectance (Pi-corrected from rhow)"
        else:
            long_name = "Remote-sensing reflectance"
        var_name = "Rrs"
        units = "sr-1"
        source_str = f"Custom Key-Matched BOA Raster Stack Collection via rioxarray"

    # Safely generate lat/lon coordinates from the uniform grid setup
    lon_grid, lat_grid = xr.broadcast(da_img.y, da_img.x)

    # Build the final data array container with grid coordinates preserved
    da_final = xr.DataArray(
        da_img.data,
        dims=("band", "y", "x"),
        coords={
            "band": da_img.band.values,
            "latitude": (("y", "x"), lat_grid.data),
            "longitude": (("y", "x"), lon_grid.data),
            # Retain the spatial projection coordinates so rioxarray understands the grid layout
            "y": da_img.y,
            "x": da_img.x,
        },
        name=var_name,
        attrs={
            "sensor": sensor,
            "long_name": long_name,
            "units": units,
            "source": source_str,
            "extent": da_img.rio.bounds(),
            "crs": str(crs_detected), # Write explicitly to attributes
        },
    )

    # Systematically write the CRS structure to the final array
    da_final = da_final.rio.write_crs(crs_detected)

    return da_final.chunk({"band": -1, "y": 100, "x": 100})  # Chunking for dask parallelism


def extract_sentinel2_default(
    file_path,
    sensor: str = "MSI",
    bands=None,
    target_resolution: int = 20,
) -> xr.DataArray:
    """
    Direct Sentinel-2 .SAFE archive ingest sub-module. Resamples all bands on-the-fly 
    to a uniform spatial resolution grid, keeping the entire pipeline fully lazy.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the root root_name.SAFE directory.
    sensor : str
        Sensor identifier platform origin string.
    bands : list or None, optional
        Target band names to isolate (e.g., ['B02', 'B03', 'B04', 'B8A']).
    target_resolution : int, optional
        The uniform output spatial resolution in meters. Default is 20.

    Returns
    -------
    da_final : xarray.DataArray
        Array with dimensions ('band', 'y', 'x') matching MDN data specifications,
        complete with projection CRS attributes.
    """
    safe_path = Path(file_path)
    
    granule_dir = safe_path / "GRANULE"
    if not granule_dir.exists():
        raise FileNotFoundError(f"Malformed .SAFE structure: Missing 'GRANULE' tracking folder.")

    # 1. Gather all .jp2 imagery files across all resolution subdirectories
    all_jp2_files = list(granule_dir.glob("**/IMG_DATA/**/*.jp2"))
    if not all_jp2_files:
        all_jp2_files = list(granule_dir.glob("**/IMG_DATA/*.jp2"))
        
    if not all_jp2_files:
        raise FileNotFoundError(f"No .jp2 imagery files found within the .SAFE folder structure.")

    # Default baseline layout bands for aquatic/environmental modeling if none provided
    if bands is None:
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    
    bands_normalized = [str(b).upper() for b in bands]
    
    # Map band names directly to their best file path match
    band_file_map = {}
    for file in all_jp2_files:
        parts = file.stem.split('_')
        band_name = parts[-2] if 'm' in parts[-1] else parts[-1]
        band_name = band_name.upper()
        
        if band_name in bands_normalized:
            if f"_{target_resolution}m" in file.name or f"R{target_resolution}m" in file.parts:
                band_file_map[band_name] = file
            elif band_name not in band_file_map:
                band_file_map[band_name] = file

    # Verify all requested bands were located
    missing_bands = [b for b in bands_normalized if b not in band_file_map]
    if missing_bands:
        raise ValueError(f"Could not locate files for the following requested bands: {missing_bands}")

    # 2. Establish the master structural template grid
    target_res_str = f"R{target_resolution}m"
    template_candidates = [f for f in band_file_map.values() if target_res_str in f.parts or f.stem.endswith(f"_{target_resolution}m")]
    template_file = template_candidates[0] if template_candidates else list(band_file_map.values())[0]
    
    da_template = xr.open_dataarray(template_file, engine="rasterio", chunks={})
    if "band" in da_template.dims:
        da_template = da_template.squeeze("band", drop=True)
        
    if not template_candidates:
        scale_factor = float(da_template.rio.transform()[0]) / target_resolution
        new_width = int(da_template.rio.width * scale_factor)
        new_height = int(da_template.rio.height * scale_factor)
        da_template = da_template.rio.reproject(
            da_template.rio.crs,
            shape=(new_height, new_width),
            resampling=Resampling.nearest
        )

    # Instead of defaulting to a specific European zone, raise an error if it fails
    if hasattr(da_template, "rio") and da_template.rio.crs is not None:
        crs_detected = da_template.rio.crs
    else:
        raise ValueError(
            f"Could not automatically detect the UTM CRS from the Sentinel-2 tile: {template_file}. "
            "Ensure rioxarray and rasterio are reading the JPEG2000 profile correctly."
        )

    # 3. Process, resample, and gather data array blocks lazily
    band_lazy_data = []
    for band_name in bands_normalized:
        file = band_file_map[band_name]
        da_band = xr.open_dataarray(file, engine="rasterio", chunks={})
        if "band" in da_band.dims:
            da_band = da_band.squeeze("band", drop=True)
            
        # Resample matching target spatial grid scale if sizes differ
        if da_band.shape != da_template.shape:
            da_band = da_band.rio.reproject_match(da_template, resampling=Resampling.nearest)
        else:
            da_band = da_band.assign_coords(x=da_template.x, y=da_template.y)
            
        band_lazy_data.append(da_band.data)

    # Stack separate arrays along a brand new leading band axis (expects dask.array imported as da)
    stacked_data = da.stack(band_lazy_data, axis=0)

    # Apply default Sentinel-2 absolute scale normalization factor (DN / 10000.0)
    scaled_data = stacked_data / 10000.0

    # Safely generate lat/lon coordinates from the template spatial grids
    lon_grid, lat_grid = xr.broadcast(da_template.y, da_template.x)

    # 5. Construct final unified DataArray container
    da_final = xr.DataArray(
        scaled_data,
        dims=("band", "y", "x"),
        coords={
            "band": bands_normalized,
            "latitude": (("y", "x"), lat_grid.data),
            "longitude": (("y", "x"), lon_grid.data),
            # Maintain structural y and x coordinates so rioxarray understands the projection mapping
            "y": da_template.y,
            "x": da_template.x,
        },
        name="reflectance",
        attrs={
            "sensor": sensor,
            "spatial_resolution": f"{target_resolution}m",
            "extent": da_template.rio.bounds(),
            "spatial_ref": str(crs_detected), # Write explicitly to attributes dict
        },
    )

    # Systematically register the spatial projection coordinate system
    da_final = da_final.rio.write_crs(crs_detected)

    return da_final.chunk({"band": -1, "y": 100, "x": 100})  # Chunking for dask parallelism


def extract_satellite_data(
    file_path: Union[str, Path],
    sensor: str,
    bands: Optional[List[Union[str, float, int]]] = None,
    json_metadata_path: Optional[Union[str, Path]] = None,
    wavelength_unit: str = "nm",
    tolerance: Optional[float] = None,
    rhow_flag: bool = False,
    flag_toa: bool = False,
    verbose: bool = False,
) -> xr.DataArray:
    """
    Unified wrapper pipeline to load and standardize satellite reflectance arrays 
    (Surface Rrs or Top-of-Atmosphere rhot) based on target filesystem structures.

    This router inspects the designated path to determine whether it points to a 
    directory stack of individual band assets, a unified multi-spectral composite 
    GeoTIFF imagery product, or a standard SeaDAS-style hierarchical Level-2 NetCDF 
    satellite file, dispatching data execution streams seamlessly.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path targeting the target spatial asset directory folder or image source file.
    sensor : str
        Sensor identifier (e.g., 'OLI', 'OLCI', 'MSI') checked against valid platforms.
    bands : list of str, float, int or None, optional
        Target band identifiers or center wavelengths to extract. Default is None.
    json_metadata_path : str, pathlib.Path or None, optional
        Path to accompanying STAC JSON metadata, required for multi-band composites.
    wavelength_unit : str, optional
        Unit configuration definition mapping for search values ('nm' vs. 'um'). 
        Default is "nm".
    tolerance : float, optional
        Acceptable coordinate search filter window width radius. Default is None.
    rhow_flag : bool, optional
        If True, and flag_toa is False, incoming surface reflectance data matrices 
        are divided by Pi where necessary to yield standardized Remote-sensing 
        reflectance properties. Default is False.
    flag_toa : bool, optional
        If True, switches processing focus to track and extract Top-of-Atmosphere 
        (TOA) reflectance parameters ('rhot') across all asset routers. Bypasses 
        surface-level Rrs scaling constraints. Default is False.
    verbose : bool, optional
        If True, broadcasts execution processing updates to stdout. Default is False.

    Returns
    -------
    da : xarray.DataArray
        A standardized multi-dimensional image array containing:
        - Dimensions: ("band", "y", "x")
        - Coordinates:
            * band: Extracted wavelength track index metrics or explicit keys.
            * latitude: 2D array matrix of shape (y, x) containing cell locations.
            * longitude: 2D array matrix of shape (y, x) containing cell locations.
        - Attributes: Standard pipeline tracking metrics (`sensor`, `long_name`, `units`, 
          `source`, bounding spatial `extent`, and projection structures).

    Raises
    ------
    FileNotFoundError
        Raised if the target file or folder path does not exist.
    ValueError
        Raised if the sensor is unsupported, if missing accompanying JSON metadata 
        arguments when processing multi-band image files, or if format structures 
        cannot be recognized.
    """
    # ------------------------------------------------------------------
    # Validate Sensor Support
    # ------------------------------------------------------------------
    if not check_sensor_availability(sensor):
        raise ValueError(
            f"Requested platform sensor '{sensor}' is currently unsupported or "
            f"not registered in the active processing environment configuration."
        )

    path_obj = Path(file_path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Provided path does not exist: {path_obj}")

    # 1. Directory Input -> Process individual band GeoTIFF stack folder
    if path_obj.is_dir():
        if verbose:
            print(f"Directory source detected. Delegating to extract_band_geotiff for sensor {sensor}: {path_obj.name}")
        return extract_band_geotiff(
            file_path=path_obj,
            bands=bands,
            sensor=sensor,
            wavelength_unit=wavelength_unit,
            tolerance=tolerance,
            rhow_flag=rhow_flag,
            flag_toa=flag_toa,
        )

    # 2. NetCDF File Input -> Process SeaDAS level-2 satellite variables
    if path_obj.suffix.lower() in [".nc", ".nc4", ".hdf", ".h5"]:
        if verbose:
            print(f"NetCDF file source detected. Delegating to extract_netcdf_satellite_rrs for sensor {sensor}: {path_obj.name}")
        return extract_netcdf_satellite_rrs(
            nc_file=path_obj,
            sensor=sensor,
            bands=bands,
            wavelength_unit=wavelength_unit,
            flag_toa=flag_toa,
            verbose=verbose,
        )

    # 3. Unified Image Input -> Process multi-band GeoTIFF composites (Requires STAC sidecar)
    if path_obj.suffix.lower() in [".tif", ".tiff"]:
        if json_metadata_path is None:
            raise ValueError(
                f"Processing a multi-band composite GeoTIFF ({path_obj.name}) requires "
                f"providing an accompanying 'json_metadata_path' STAC sidecar asset map."
            )
        if verbose:
            print(f"Composite GeoTIFF source detected. Delegating to extract_composite_geotiff_wjson for sensor {sensor}: {path_obj.name}")
        return extract_composite_geotiff_wjson(
            file_path=path_obj,
            json_metadata_path=json_metadata_path,
            sensor=sensor,
            bands=bands,
            wavelength_unit=wavelength_unit,
            tolerance=tolerance,
            rhow_flag=rhow_flag,
            flag_toa=flag_toa,
        )

    raise ValueError(
        f"Unsupported file format file structure signature: '{path_obj.suffix}'. "
        f"Unable to route target data handling pipelines automatically."
    )