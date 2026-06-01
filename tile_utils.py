import numpy as np
from pathlib import Path
import rasterio
from functools import lru_cache
from netCDF4 import Dataset
from pyproj import Proj
import sys
import warnings
warnings.filterwarnings('ignore')



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