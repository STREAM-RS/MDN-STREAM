# -*- coding: utf-8 -*-
"""
File Name:              user_utilities.py
Description:            This code file will be used write functions that simplify the application of the code for the endusers. It 
                        will be used to abstract detail sections like argument setting etc completely away from the end users

Date Created:           April 17th, 2026
Author:                 Arun M Saranathan
Email:                  arun.saranathan@ssaihq.com/
                        fnu.arunmuralidharansaranathan@nasa.gov
"""


import numpy as np
import xarray as xr
import pandas as pd
import dask
from dask.diagnostics import ProgressBar
from dask import array as da
import tensorflow as tf

from typing import Optional, Tuple, Union, List, Dict
import re
import gc

from .meta import get_sensor_bands
from .parameters import get_args
from .utilities import get_mdn_predictions_and_uncertainties, get_mdn_preds_raw, get_mdn_preds_uncertainties
from .utils import mask_land

#rgb_bands = [660, 550, 440]
min_in_out_val = 1e-6

# If new default model is defined for a sensor this Dictionary needs to be updated.
DEFAULT_SENSOR_PRODUCT_COMBINATIONS = {
    "MSI"         : 'chl,tss,cdom',
    "OLI"         : 'chl,tss,cdom',
    "OLCI"         : 'chl,tss,cdom,pc',
    "PACE-delivery": 'aph,chl,tss,pc,ad,ag,cdom',
    "EMIT"         : 'aph,chl,tss,pc,ad,ag,cdom',
    "PRISM"        : 'aph,chl,tss,pc,ad,ag,cdom',
    "AVIRISNG"     : 'aph,chl,tss,pc,ad,ag,cdom',
    "PACE-delivery": 'aph,chl,tss,pc,ad,ag,cdom',
    "PRISMA"       : 'aph,chl,tss,pc,ad,ag,cdom',
    "HICO"         : 'aph,chl,tss,pc,ad,ag,cdom',
    "MOD"          : 'Chl,TSS,aCDOM443,aCDOM555,aNAP443,aNAP555,aph443,aph488,aph555,aph667',
    "VI"           : 'Chl,TSS,aCDOM443,aCDOM555,aNAP443,aNAP555,aph443,aph488,aph555,aph667',
    "MERIS"        : 'Chl,TSS,aCDOM443,aCDOM555,aNAP443,aNAP555,aph443,aph488,aph555,aph667',
    "SD8-cc_base"  : 'chl,secchi',
}
PRODUCT_PATTERN=  r'^[^\s,]+(,[^\s,]+)+$'


def get_default_pipeline_kwargs(sensor, product):
    """
    This function will return the default arguments of the MDN package for a specific sensor-product combination. If a default version does
    not exist it will throw an error.

    Parameters
    ----------
    sensor : str
        Sensor name. Must be valid sensor defined in meta.py and supported by the package

    product: str
        Comma-separated string listing all the products. Only supports default versions as defined here.


    Returns
    -------
    kwargs: dict
        A dictionary containing the default arguments needed to run an MDN model for the specific sensor-product combination. 

    """
    
    # Helper to check if CSV string is subset of another
    def is_subset_product(sub, full):
        sub_set = {s.strip() for s in sub.split(',')}
        full_set = {f.strip() for f in full.split(',')}
        return sub_set.issubset(full_set)

    # Validate sensor exists
    supported_sensors = list(DEFAULT_SENSOR_PRODUCT_COMBINATIONS.keys())
    assert sensor in supported_sensors, (
        f"The sensor: {sensor} is not currently supported. "
        f"The supported sensors are: {supported_sensors}"
    )

    kwargs = None

    if sensor in ["MOD", 'MODA', 'MODT']:
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
        kwargs = {
           'product': "Chl,TSS,aCDOM443,aCDOM555,aNAP443,aNAP555,aph443,aph488,aph555,aph667",
           'sat_bands': False,
           'model_loc': "Weights",
           'sensor': "MOD",
           'silent': True,
           'model_uid': "16a3fb66d506d555444846b40f680ee76bfa36c7514940ab85b83884548e350c"
       }
    if sensor in ["OLI"]:
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
        kwargs = {
           'product': "chl,tss,cdom",
           'sat_bands': False,
           'model_loc': "Weights",
           'sensor': "OLI",
           'silent': True,
           'model_uid': "c42180fb52344bc80726538ecdfb07569b29cc802f27ab3c2789ab41302d701e"
       }
    if sensor in ["MSI"]:
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
        kwargs = {
           'product': "chl,tss,cdom",
           'sat_bands': False,
           'model_loc': "Weights",
           'sensor': "MSI",
           'silent': True,
           'model_uid': "1b79098defa9693eb03ac734467c3854cf4fb81ce3d1a96b9bf0f511f1c997b1"
       }
    if sensor in ["VI"]:
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
        kwargs = {
           'product': "Chl,TSS,aCDOM443,aCDOM555,aNAP443,aNAP555,aph443,aph488,aph555,aph667",
           'sat_bands': False,
           'model_loc': "Weights",
           'sensor': "VI",
           'silent': True,
           'model_uid': "6f7262564d6e2a2deb3b3e433384f60398acc38e3a9712563e70445bc80ab924"
       }
    if sensor in ["MERIS"]:
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
        kwargs = {
           'product': "Chl,TSS,aCDOM443,aCDOM555,aNAP443,aNAP555,aph443,aph488,aph555,aph667",
           'sat_bands': False,
           'model_loc': "Weights",
           'sensor': "MERIS",
           'silent': True,
           'model_uid': "fadf7c51442969f4f9cde83fdad8a1b958a1dff1523f66fc331f376877e02b59"
       }
    if sensor in ["HICO"]:
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
        kwargs = {
           'product': "chl",
           'sat_bands': False,
           'model_loc': "Weights",
           'sensor': "HICO",
           'silent': True,
           'model_uid': "fa50ae29755c1f2cf7b3eaeb552245830c580cbb59b0b9e39992ff89c0cc3020"
       }
    # Logic for OLCI Sensor
    if sensor in ["S3A", 'S3B', 'OLCI']:
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
        if product == "chl":
            kwargs = {
                'product': "chl",
                'sat_bands': False,
                'model_loc': "Weights_test",
                'sensor': "OLCI",
                'silent': True,
                'model_uid': "39863a30bd3ea0c25f24a212564810cfc341ca66b6c10c8b464befac7fbf6a8f"
            }
        elif product=="chl,tss,cdom":
            kwargs = {
                'product': max_model_products,
                'sat_bands': False,
                'model_loc': "Weights",
                'sensor': "OLCI",
                'silent': True,
                'model_uid': "73bf3ca36f95d13a38032a36f7565a992fa772af0833ad2f74b710b6df33eba2"
            }
        elif is_subset_product(product, max_model_products):
            kwargs = {
                'product': max_model_products,
                'sat_bands': False,
                'model_loc': "Weights",
                'sensor': 'OLCI',
                'silent': True,
                'model_uid': "73bf3ca36f95d13a38032a36f7565a992fa772af0833ad2f74b710b6df33eba2"
            }
        else:
            raise ValueError(
                f"No model found that supports {product} for the {sensor} sensor. "
                f"The most diverse model only supports: {max_model_products}"
            )

    # Logic for Planet super-dove Sensor
    elif sensor == "SD8-cc_base":
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
        if is_subset_product(product, max_model_products):
            kwargs = {
                'product': max_model_products,
                'model_loc': "Weights",
                'sat_bands': False,
                'sensor': sensor,
                'silent': True,
                'model_uid': "69fee32c5fe248a5390f83b3eef2e4230d3f1e85507abaea670a4a8b448a6f8d",
            }
        else:
            raise ValueError(
                f"No model found that supports {product} for the {sensor} sensor. "
                f"The most diverse model only supports: {max_model_products}"
            )
    
    
    # Logic for PACE-delivery Sensor
    elif sensor in ["PACE-delivery","EMIT","AVIRISNG","PRISM","PRISMA"]: #"HICO",
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
        model_uid_dict = {
                            "PACE-delivery":"6f2a6b07f6e8b5723a80c389456e13a6f17d7db02024a425f15f0b340fbb97e0",
                            "EMIT"         :"a3fef49195d5c65f17ca3b34ba6563a41879aa7e15685b18d646d890f7c282b2",
                            "PRISM"        :"4e4cecb75957e060e0caf70f751e6435dad5e391d04485f5376f061081eeec67",
                            "AVIRISNG"     :"87e72ac535fd367718e019c3671deb665a999b5550a9caeb343a681afdf710a9",
                            "PRISMA"       :"6041caec3d8c34771f9082740fc3cee1a16d3b1b21cfda0f245e615a0a01570d",
                            "HICO"         :"b978ee38b759569c6c6860a6a875f23131cef5ecf5b93d0d67708cb408718c85",
                            }
        if is_subset_product(product, max_model_products):
            kwargs = {
                'allow_missing': False,
                'allow_nan_inp': False,
                'allow_nan_out': True,
                'sensor': sensor,
                'removed_dataset': "South_Africa", # simplified from the previous conditional logic
                'filter_ad_ag': False,
                'imputations': 5,
                'no_bagging': False,
                'plot_loss': False,
                'benchmark': False,
                'sat_bands': False,
                'n_iter': 31622,
                'n_mix': 5,
                'n_hidden': 446,
                'n_layers': 5,
                'lr': 1e-3,
                'l2': 1e-3,
                'epsilon': 1e-3,
                'batch': 128,
                'use_HICO_aph': True,
                'n_rounds': 10,
                'product': max_model_products,
                'use_gpu': False,
                'data_loc': "/home/ryanoshea/in_situ_database/Working_in_situ_dataset/Augmented_Gloria_V3_2/",
                'use_ratio': True,
                'min_in_out_val': min_in_out_val, 
                'silent': True,
                'no_data': -999.,
                'model_uid': model_uid_dict[sensor],
            }

            # Append wavelength metadata
            kwargs.update({
                'aph_wavelengths': get_sensor_bands(kwargs['sensor'] + '-aph'),
                'adag_wavelengths': get_sensor_bands(kwargs['sensor'] + '-adag'),
            })
            
            if not kwargs['benchmark']:
                kwargs['plot_loss'] = False
        else:
            raise ValueError(
                f"No model found that supports {product} for the {sensor} sensor. "
                f"The most diverse model only supports: {max_model_products}"
            )

    return kwargs


def get_spectral_preds_raw(
        test_x: np.ndarray,
        sensor: str = "OLCI",
        products: str = "chl",
    ) -> Tuple[dict, dict, slice]:
    """
    Generate MDN predictions for a given spectral input (2D) dataset from the default model

    Parameters
    ----------
    test_x : np.ndarray
        Input data (n_samples x n_features).

    sensor : str
        Sensor name ).

    products : str
        Products to predict.

   Returns
    -------
    output : np.ndarray
        Predictions (shape depends on mode):
        - "point": (n_samples, n_outputs)
        - "full": (n_models, n_samples, n_outputs)

    [uncertainties] : dict  [Optional, based on return_uncert]
        - composite mode: {'comp_unc': ...}
        - limits mode: {'low_lim': ..., 'high_lim': ...}

    op_slices : dict
        Dictionary of output slices per predicted product.

    """

    # First get the arguments for the default model
    kwargs = get_default_pipeline_kwargs(sensor=sensor, product=products)
    args = get_args(**kwargs)

    # Get the predictions from the MDN
    outputs, op_slices = get_mdn_preds_raw(test_x, args=args, op_mode="full", scaler_mode="non_invert")

    _ , uncertainties = get_mdn_predictions_and_uncertainties(mdn_outputs=outputs['coefs'],  op_mode="full", scaler_mode="non_invert", uncert_mode="composite")                                                                 

    return outputs, uncertainties, op_slices


def subset_mdn_by_variable_slices(
        mdn_preds: Union[Dict[str, np.ndarray], np.ndarray],
        mdn_uncert: Union[Dict[str, np.ndarray], np.ndarray],
        mdn_preds_slices: Dict[str, Union[slice, tuple, list, int]],
        target_keys: List[str]
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], Union[Dict[str, np.ndarray], np.ndarray], dict]:
    """
    Filters MDN prediction and uncertainty structures along their last axis 
    using a set of target keys, and recalculates their relative output slices.

    Parameters
    ----------
    mdn_preds : dict or np.ndarray
        Predictions structure. If a array, shape is typically 
        (n_models, n_samples, n_outputs). If a dictionary, values are arrays.

    mdn_uncert : dict or np.ndarray
        Uncertainties structure matching the type and shape behavior of mdn_preds.

    mdn_preds_slices : dict
        Dictionary mapping original feature keys to their index or slice bounds 
        along the last axis (n_outputs).

    target_keys : list of str
        The specific product/feature keys to extract from the datasets.

    Returns
    -------
    updated_preds : dict or np.ndarray
        Subsetted predictions containing only columns belonging to target_keys.

    updated_uncert : dict or np.ndarray
        Subsetted uncertainties containing only columns belonging to target_keys.

    updated_slices : dict
        New dictionary of output slices adjusted relative to the newly shifted 
        and sequential output array columns.
    """
    valid_keys = [k for k in target_keys if k in mdn_preds_slices]
    if not valid_keys:
        raise ValueError(f"None of the target keys {target_keys} exist in the model's output slices: {list(mdn_preds_slices.keys())}")
        
    column_indices = []
    updated_slices = {}
    current_new_idx = 0
    
    for key in valid_keys:
        orig_slice = mdn_preds_slices[key]
        
        # Convert slice, tuple, list, or single integer to a flat list of column indices
        if isinstance(orig_slice, slice):
            start = orig_slice.start if orig_slice.start is not None else 0
            stop = orig_slice.stop
            feature_indices = list(range(start, stop))
        elif isinstance(orig_slice, (tuple, list)):
            feature_indices = list(orig_slice)
        else:
            feature_indices = [orig_slice]
            
        column_indices.extend(feature_indices)
        
        # Track the new shifted slice bounds
        feature_width = len(feature_indices)
        updated_slices[key] = slice(current_new_idx, current_new_idx + feature_width)
        current_new_idx += feature_width

    # Extract along the very last axis using the accumulated column indices
    if isinstance(mdn_preds, dict):
        updated_preds = {k: v[..., column_indices] for k, v in mdn_preds.items() if v.ndim == 3}
        updated_uncert = {k: v[..., column_indices] for k, v in mdn_uncert.items() if v.ndim == 3}
    else:
        updated_preds = mdn_preds[..., column_indices]
        updated_uncert = mdn_uncert[..., column_indices]

    # Pass over the selected_index if it exists in the original predictions (only needed for the "select" mode)
    if 'selected_index' in mdn_preds:
        updated_preds['selected_index'] = mdn_preds['selected_index']
        
    return updated_preds, updated_uncert, updated_slices


def get_spectral_preds(
        test_x: np.ndarray,
        sensor: str = "OLCI",
        products: str = "chl",
        op_mode: str = "select",
        return_uncert: bool = True,
        uncert_mode: str = "limits",
        scaler_mode: str = "invert",
        progress_vis: bool = True
    ) -> Union[Tuple[Union[Dict[str, np.ndarray], np.ndarray], Union[Dict[str, np.ndarray], np.ndarray], dict], Tuple[Union[Dict[str, np.ndarray], np.ndarray], dict]]:
    """
    Generate MDN predictions for a given spectral input (2D) dataset from the default model.

    Parameters
    ----------
    test_x : np.ndarray
        Input data (n_samples x n_features).

    sensor : str
        Sensor name (e.g., "OLCI").

    products : str
        Comma-separated products to predict and extract (e.g., "chl,aph").

    op_mode : str
        Operation mode controlling network processing dimensions (e.g., "select").

    return_uncert : bool
        Whether the function returns the uncertainties. (Default: True)

    uncert_mode : {"composite", "limits"}
        Defines the mode in which the uncertainty is returned:
            - "limits": returns the upper and lower limits as estimated from the predicted distribution
            - "composite": returns the average distance on each side

    scaler_mode : {"invert", "non_invert"}
        Whether to apply inverse scaling to outputs. (Default: invert)

    progress_vis : bool
        Whether the progress of the tqdms are shown on screen. (Default: True)

    Returns
    -------
    estimates : dict or np.ndarray
        Predictions containing only columns belonging to the requested products.
        Shapes are preserved from underlying predictor depending on op_mode.

    [uncert] : dict or np.ndarray [Optional, based on return_uncert]
        Uncertainties structure containing only columns belonging to requested products.

    selected_slices : dict
        Dictionary of updated output slices mapped per predicted product.
    """

    # 1. Fetch the default pipeline arguments
    kwargs = get_default_pipeline_kwargs(sensor=sensor, product=products)
    args = get_args(**kwargs)

    # 2. Call the base predictor function
    mdn_preds, mdn_uncert, mdn_preds_slices = get_mdn_preds_uncertainties(
        test_x=test_x, args=args, op_mode=op_mode, 
        scaler_mode=scaler_mode, uncert_mode=uncert_mode, progress_vis=progress_vis
    )

    # 3. Parse the requested target products
    target_products = [s.strip() for s in products.split(',')]
    
    # 4. Extract target variables and recalculate shifting slices
    estimates, uncert, selected_slices = subset_mdn_by_variable_slices(
        mdn_preds, mdn_uncert, mdn_preds_slices, target_products
    )
    
    # 5. Structured Return
    if return_uncert:
        return estimates, uncert, selected_slices
    else:
        return estimates, selected_slices



def map_cube_mdn_chunk(
    img_chunk: xr.DataArray,
    args: Dict,
    target_products: str = "chl",
    n_outputs: int = 1,
    wvl_bands: Union[List[float], np.ndarray] = None,
    op_mode: str = "select",
    return_uncert: bool = True,
    scaler_mode: str = "invert",
    land_mask: bool = False,
    landmask_threshold: float = 0.0,
) -> xr.Dataset:
    """
    Map a chunk of an image cube using MDN and return an xarray Dataset.

    Parameters
    ----------
    img_chunk : xr.DataArray
        An individual localized chunk slice of the input satellite image.
    args : Dict
        Configuration dictionary containing pipeline running options.
    target_products : str, default 'chl'
        A comma-separated string indicating target water constituents to calculate.
    n_outputs : int, default 1
        Calculated output array depth (matching the split-length of target_products).
    wvl_bands : Union[List[float], np.ndarray], default None
        Wavelength arrays configured for the target sensor.
    op_mode : str, default 'select'
        Operation mode parameter ('select' or 'ensemble').
    return_uncert : bool, default True
        If True, calculates uncertainty ranges dynamically.
    scaler_mode : str, default 'invert'
        Scaler mode config passed to the get_spectral_preds pipeline helper.
    land_mask : bool, default False
        If True, calculates a spatial land mask inside this chunk block.
    landmask_threshold : float, default 0.0
        Wavelength threshold utilized by the land masking algorithm.

    Returns
    -------
    xr.Dataset
        An xr.Dataset block representing processed output variables for this chunk area.
    """
    # 1. Transpose chunk and extract NumPy structures safely
    img_chunk_t = img_chunk.transpose("y", "x", "band")
    img_chunk_np = np.asarray(img_chunk_t.data)  # Safely coerces the Dask chunk to a local NumPy array

    # 2. Extract metadata parameters safely
    n_rounds = (args.get("n_rounds", args["n_rounds"])
        if isinstance(args, dict)
        else args.n_rounds
    )
    n_models = 1 if op_mode == "select" else n_rounds
    no_data_val = (args.get("no_data_val", args.get("no_data", -9999.0))
        if isinstance(args, dict)
        else getattr(args, "no_data_val", getattr(args, "no_data", -9999.0))
    )

    # -------------------------------------------------------------------------
    # In-place Array Cleaning on img_chunk_np
    # -------------------------------------------------------------------------
    # Drop +/- Inf values up front
    img_chunk_np[np.isinf(img_chunk_np)] = np.nan

    # Replace negative values with no_data_val directly on img_chunk_np
    img_chunk_np[img_chunk_np < 0] = no_data_val

    # 3. Initialize blank output structures using the correct 'no_data' value
    img_preds = no_data_val * np.ones(
        (n_models, img_chunk_np.shape[0], img_chunk_np.shape[1], n_outputs),
        dtype=np.float32,
    )

    if return_uncert:
        img_uncert_lb = no_data_val * np.ones(( n_models, img_chunk_np.shape[0], img_chunk_np.shape[1], n_outputs,), dtype=np.float32,)
        img_uncert_ub = no_data_val * np.ones(( n_models, img_chunk_np.shape[0], img_chunk_np.shape[1], n_outputs,), dtype=np.float32,)

    #  Create water mask
    if land_mask:
        img_mask = mask_land(img_chunk_np, wvl_bands, threshold=landmask_threshold)
    else:
        # img_mask = np.isnan(np.min(img_chunk_np, axis=2)).astype(float)
        img_mask = np.all(np.isnan(img_chunk_np), axis=2).astype(float)

    bool_mask = img_mask == 0
    water_pixels = np.where(bool_mask)
    water_spectra = img_chunk_np[bool_mask]

    # First filter: Remove spectra with majority invalid/don't-care values
    maj_neg = ((water_spectra == no_data_val) | (water_spectra < 1e-6)).sum(axis=1) > 5
    water_spectra = water_spectra[~maj_neg]
    water_pixels = tuple(p[~maj_neg] for p in water_pixels)

    # Second filter: Prepare spectra and drop any remaining NaNs
    water_final = np.ma.masked_invalid(water_spectra).reshape((-1, water_spectra.shape[-1]))
    valid_mask = ~np.any(water_final.mask, axis=1)

    water_final = water_final[valid_mask]
    water_pixels = tuple(p[valid_mask] for p in water_pixels)

    if isinstance(water_final, np.ma.MaskedArray):
        water_final = water_final.filled(no_data_val)

    # Also since the MDN expects positive values replace negatives with 1.e-6
    water_final[water_final < 1.e-6] = 1.e-6

    # Build dynamic output dict base
    data_vars = {"predictions": (("model", "y", "x", "output"), img_preds)}

    # If we have valid water pixels, process them through the MDN model
    if water_final.size > 0:
        sensor_str = args["sensor"] if isinstance(args, dict) else args.sensor

        if return_uncert:
            preds, uncert, _ = get_spectral_preds(
                test_x=water_final,
                sensor=sensor_str,
                products=target_products,
                op_mode=op_mode,
                return_uncert=return_uncert,
                uncert_mode="limits",
                progress_vis=False,
            )
        else:
            preds, _ = get_spectral_preds(
                test_x=water_final,
                sensor=sensor_str,
                products=target_products,
                op_mode=op_mode,
                return_uncert=return_uncert,
                uncert_mode="limits",
                progress_vis=False,
            )

        # Re-assign the predictions back into the spatial grid coordinate slices
        img_preds[:, water_pixels[0], water_pixels[1], :] = preds["pred"]
        data_vars["predictions"] = (("model", "y", "x", "output"), img_preds)

        if return_uncert:
            img_uncert_lb[:, water_pixels[0], water_pixels[1], :] = uncert[
                "low_lim"
            ]
            img_uncert_ub[:, water_pixels[0], water_pixels[1], :] = uncert[
                "high_lim"
            ]

    if return_uncert:
        data_vars["uncertainty_low"] = (
            ("model", "y", "x", "output"),
            img_uncert_lb,
        )
        data_vars["uncertainty_high"] = (
            ("model", "y", "x", "output"),
            img_uncert_ub,
        )

    # =================================================================
    # CRITICAL FIX: DO NOT DEFINE SPATIAL COORDINATES HERE AT ALL
    # =================================================================
    ds_chunk = xr.Dataset(
        data_vars=data_vars,
        coords={
            "model": np.arange(n_models),
            "output": np.arange(n_outputs),
        },
    )

    # Explicitly clear temporary arrays before returning
    del img_chunk_np, water_spectra, water_final
    gc.collect()
    tf.keras.backend.clear_session()

    return ds_chunk


def map_cube_mdn(
    img_data: xr.DataArray,
    sensor: str,
    products: str = "chl",
    op_mode: str = "select",
    return_uncert: bool = True,
    uncert_mode: str = "limits",
    scaler_mode: str = "invert",
    land_mask: bool = False,
    landmask_threshold: float = 0.0,
    progress_vis: bool = True,
) -> xr.Dataset:
    """
    Map an entire image cube using MDN via parallelized Dask block execution.

    Parameters
    ----------
    img_data : xr.DataArray
        The input satellite image cube containing spatial dimensions (y, x) and bands.
    sensor : str
        The satellite sensor name (e.g., 'MSI', 'OLCI', 'OLI').
    products : str, default 'chl'
        Comma-separated string of target products to retrieve (e.g., 'chl,tss').
    op_mode : str, default 'select'
        Operation mode for model execution (e.g., 'select' or 'ensemble').
    return_uncert : bool, default True
        If True, returns low and high uncertainty limits alongside predictions.
    uncert_mode : str, default 'limits'
        The format of uncertainty estimations (e.g., 'limits').
    scaler_mode : str, default 'invert'
        The mode used by scaling pipelines when preprocessing/postprocessing values.
    land_mask : bool, default False
        If True, applies a band-based thresholding filter to mask land pixels.
    landmask_threshold : float, default 0.0
        The threshold value utilized during the land masking routine.
    progress_vis : bool, default True
        If True, shows a visual Dask ProgressBar tracking processing chunks.

    Returns
    -------
    xr.Dataset
        The computed dataset containing output predictions and uncertainty estimations.
    """
    #  Fetch pipeline configurations
    kwargs = get_default_pipeline_kwargs(sensor=sensor, product=products)
    args = get_args(**kwargs)
    
    #  Extract configuration metadata
    wvl_bands = get_sensor_bands(sensor)
    n_rounds = args.get("n_rounds", args["n_rounds"]) if isinstance(args, dict) else args.n_rounds
    # n_outputs = len(products.split(","))
    n_models = 1 if op_mode == "select" else n_rounds
    no_data_val = args.get("no_data", args["no_data"]) if isinstance(args, dict) else args.no_data

    # Fetch op_slices once globally using a dummy single pixel vector
    dummy_pixel = 0.01 * np.ones((1, len(wvl_bands)))
    _, op_slices =  get_spectral_preds(test_x=dummy_pixel, sensor= sensor, products=products, op_mode=op_mode, return_uncert=False, uncert_mode="limits",
                                        progress_vis=False) 

    # Find the length of the output vector
    n_outputs = 0
    for key in op_slices:
        orig_slice = op_slices[key]
        
        # Convert slice, tuple, list, or single integer to a flat list of column indices
        if isinstance(orig_slice, slice):
            start = orig_slice.start if orig_slice.start is not None else 0
            stop = orig_slice.stop
            feature_indices = list(range(start, stop))
        elif isinstance(orig_slice, (tuple, list)):
            feature_indices = list(orig_slice)
        else:
            feature_indices = [orig_slice]
        
        # Track the new shifted slice bounds
        n_outputs += len(feature_indices)

    # Create dummy arrays for building the template
    dummy_preds = np.empty((n_models, len(img_data.y), len(img_data.x), n_outputs), dtype=np.float32)
    
    template_vars = {
        "predictions": (("model", "y", "x", "output"), dummy_preds)
    }
    
    if return_uncert:
        dummy_uncert = np.empty((n_models, len(img_data.y), len(img_data.x), n_outputs), dtype=np.float32)
        template_vars["uncertainty_low"] = (("model", "y", "x", "output"), dummy_uncert)
        template_vars["uncertainty_high"] = (("model", "y", "x", "output"), dummy_uncert)

    # 4. Define the initial full template dataset
    template_ds = xr.Dataset(
        data_vars=template_vars,
        coords={
            "model": np.arange(n_models),
            "y": img_data.y,
            "x": img_data.x,
            "output": np.arange(n_outputs),
        }
    )
    if "spatial_ref" in img_data.coords:
        template_ds = template_ds.assign_coords({"spatial_ref": img_data.spatial_ref})

    # =================================================================
    # CRITICAL FIX: PRESERVE, STRIP & CHUNK COORDINATES
    # =================================================================
    # Capture the global coordinates to safely bind back on later
    global_coords = {
        "y": img_data.coords["y"],
        "x": img_data.coords["x"]
    }
    if "spatial_ref" in img_data.coords:
        global_coords["spatial_ref"] = img_data.coords["spatial_ref"]
    if "latitude" in img_data.coords:
        global_coords["latitude"] = img_data.coords["latitude"]
    if "longitude" in img_data.coords:
        global_coords["longitude"] = img_data.coords["longitude"]

    # Strip coordinates from template so xr.map_blocks doesn't perform strict coordinate validation
    coords_to_drop = ["y", "x", "spatial_ref", "latitude", "longitude"]
    clean_template = template_ds.drop_vars(coords_to_drop, errors="ignore")

    # Chunk the template matching the input image chunks so it has Dask arrays
    clean_template = clean_template.chunk({
        "y": img_data.chunksizes["y"],
        "x": img_data.chunksizes["x"]
    })

    # 5. Run mapping across dask-chunk blocks lazily
    lazy_result_ds = xr.map_blocks(
        map_cube_mdn_chunk,
        img_data, 
        kwargs={
            "args": args,
            "target_products": products,
            "n_outputs": n_outputs,
            "wvl_bands": wvl_bands,
            "op_mode": op_mode,
            "land_mask": land_mask,
            "landmask_threshold": landmask_threshold,
            "scaler_mode": scaler_mode,
            "return_uncert": return_uncert
        },
        template=clean_template  # Use clean coordinate-free chunked template
    )

    # Re-assign the true global coordinates back to the lazy dataset before computing
    final_lazy_ds = lazy_result_ds.assign_coords(global_coords)

    # Execute computation with visual feedback
    if progress_vis:
        from dask.diagnostics import ProgressBar
        print("Computing MDN predictions across chunks...")
        with ProgressBar():
            final_ds = final_lazy_ds.compute(scheduler="single-threaded")
    else:
        final_ds = final_lazy_ds.compute(scheduler="single-threaded")

    # EXTRACT SPATIAL EXTENT AS A TUPLE
    lon_min = float(final_ds.coords["longitude"].min())
    lon_max = float(final_ds.coords["longitude"].max())
    lat_min = float(final_ds.coords["latitude"].min())
    lat_max = float(final_ds.coords["latitude"].max())

    # Get the wavelengths of spectral products
    aph_wvl, adag_wvl = np.asarray([]), np.asarray([])
    for sp_prod in ["aph", "ad", "ag"]:
        if sp_prod in products:
            if sp_prod == "aph":
                aph_wvl = np.asarray(get_sensor_bands((sensor.split("-")[0] + '-aph')))
            else:
                adag_wvl= np.asarray(get_sensor_bands((sensor.split("-")[0] + '-aph')))
                break                          # need to check only one of ad or ag


    # ADD METADATA ATTRIBUTES HERE
    final_ds.attrs.update({
        "title": "MDN Satellite Product Predictions",
        "sensor": sensor,
        "target_products": products,
        "operation_mode": op_mode,
        "land_mask_applied": str(land_mask),
        "land_mask_threshold": landmask_threshold,
        "scaler_mode": scaler_mode,
        "no_data_value": no_data_val,
        "extent": [lon_min, lon_max, lat_min, lat_max],
        "history": f"Created on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} using map_cube_mdn"
    })

    # IF NEEDED add the wavelengths associated with the spectral components
    if aph_wvl.size !=0:
        final_ds.attrs["aph_wavelengths"] = aph_wvl.tolist()

    if adag_wvl.size !=0:
        final_ds.attrs["adag_wavelengths"] = adag_wvl.tolist()

    return final_ds, op_slices