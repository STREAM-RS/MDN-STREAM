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
import dask
from dask.diagnostics import ProgressBar

from typing import Optional, Tuple, Union, List, Dict
import re

from .meta import get_sensor_bands
from .parameters import get_args
from .utilities import get_mdn_predictions_and_uncertainties, get_mdn_preds_raw, get_mdn_preds_uncertainties
from .utils import mask_land

#rgb_bands = [660, 550, 440]
min_in_out_val = 1e-6

# If new default model is defined for a sensor this Dictionary needs to be updated.
DEFAULT_SENSOR_PRODUCT_COMBINATIONS = {
    "OLCI": 'chl,tss,cdom',
    "PACE-delivery": 'aph,chl,tss,pc,ad,ag,cdom',
    "SD8-cc_base": 'chl,secchi',
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

    # Logic for OLCI Sensor
    if sensor == "OLCI":
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
        if product == "chl":
            kwargs = {
                'product': "chl",
                'sat_bands': False,
                'model_loc': "Weights_test",
                'sensor': sensor,
                'model_uid': "39863a30bd3ea0c25f24a212564810cfc341ca66b6c10c8b464befac7fbf6a8f"
            }
        elif is_subset_product(product, max_model_products):
            kwargs = {
                'product': max_model_products,
                'sat_bands': False,
                'model_loc': "Weights",
                'sensor': sensor,
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
                'model_uid': "69fee32c5fe248a5390f83b3eef2e4230d3f1e85507abaea670a4a8b448a6f8d",
            }
        else:
            raise ValueError(
                f"No model found that supports {product} for the {sensor} sensor. "
                f"The most diverse model only supports: {max_model_products}"
            )
    
    
    # Logic for PACE-delivery Sensor
    elif sensor == "PACE-delivery":
        max_model_products = DEFAULT_SENSOR_PRODUCT_COMBINATIONS[sensor]
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
                'model_uid': "6f2a6b07f6e8b5723a80c389456e13a6f17d7db02024a425f15f0b340fbb97e0",
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
    args: Dict,
    img_chunk: xr.DataArray,
    target_products: str = "chl",
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
    args : dict
        Dictionary of arguments for the MDN model, typically obtained from get_default_pipeline_kwargs().
    img_chunk : xarray.DataArray
        Chunk of the image cube (nRow x nCols x nBands).
    target_products : str
        Comma-separated products to predict and extract (e.g., "chl,aph").
    wvl_bands : array-like
        Wavelengths corresponding to img_data bands.
    op_mode : {"select", "full"}
        Whether to select the median model or return full ensemble.
    return_uncert : bool
        Whether to return uncertainties.
    scaler_mode : {"invert", "non_invert"}
        Whether to invert scaled predictions.
    land_mask : bool
        Apply heuristic land masking.
    landmask_threshold : float
        Threshold for land mask.

    Returns
    -------
    ds_chunk : xarray.Dataset
        A dataset containing spatial predictions and uncertainty limits mapped for the block.
    """
    # 1. Transpose the chunk to (y, x, band) and extract NumPy arrays safely
    img_chunk_t = img_chunk.transpose("y", "x", "band")
    img_chunk_np = img_chunk_t.values

    # 2. Extract metadata parameters safely
    n_rounds = args.get("n_rounds", args["n_rounds"]) if isinstance(args, dict) else args.n_rounds
    n_outputs = len(target_products.split(","))
    n_models = 1 if op_mode == "select" else n_rounds
    no_data_val = args.get("no_data", args["no_data"]) if isinstance(args, dict) else args.no_data

    # 3. Initialize blank output structures using the correct 'no_data' value
    img_preds = no_data_val * np.ones((n_models, img_chunk_np.shape[0], img_chunk_np.shape[1], n_outputs))
    # Also initialize uncertainty grids if return_uncert is requested
    if return_uncert:
        img_uncert_lb = no_data_val * np.ones((n_models, img_chunk_np.shape[0], img_chunk_np.shape[1], n_outputs))
        img_uncert_ub = no_data_val * np.ones((n_models, img_chunk_np.shape[0], img_chunk_np.shape[1], n_outputs))
    
    # 4. Create water mask
    if land_mask:
        img_mask = mask_land(img_chunk_np, wvl_bands, threshold=landmask_threshold)
    else:
        img_mask = np.isnan(np.min(img_chunk_np, axis=2)).astype(float)

    bool_mask = (img_mask == 1)
    water_pixels = np.where(bool_mask)
    water_spectra = img_chunk_np[bool_mask]

    # First filter: Remove spectra with majority invalid/negative values
    maj_neg = (water_spectra < 1e-4).sum(axis=1) > 5
    water_spectra = water_spectra[~maj_neg]
    water_pixels = tuple(p[~maj_neg] for p in water_pixels)

    # Second filter: Prepare spectra and drop any remaining NaNs/Infs 
    water_final = np.ma.masked_invalid(water_spectra).reshape((-1, water_spectra.shape[-1]))
    valid_mask = ~np.any(water_final.mask, axis=1)
    
    water_final = water_final[valid_mask]
    water_pixels = tuple(p[valid_mask] for p in water_pixels)

    # Build dynamic output dict base
    data_vars = {
        "predictions": (("model", "y", "x", "output"), img_preds.astype(np.float32))
    }

    # If we have valid water pixels, process them through the MDN model
    if water_final.size > 0:
        if return_uncert:
            preds, uncert, _ = get_spectral_preds(test_x=water_final, sensor=args.sensor, products=target_products, op_mode=op_mode,
            return_uncert=return_uncert, uncert_mode="limits", progress_vis=False)            
        else:
            preds, _ = get_spectral_preds(test_x=water_final, sensor=args.sensor, products=target_products, op_mode=op_mode,
            return_uncert=return_uncert, uncert_mode="limits", progress_vis=False) 

        # Re-assign the predictions back into the spatial grid coordinate slices
        img_preds[:, water_pixels[0], water_pixels[1], :] = preds['pred']
        data_vars["predictions"] = (("model", "y", "x", "output"), img_preds.astype(np.float32))
        
        if return_uncert:
            img_uncert_lb[:, water_pixels[0], water_pixels[1], :] = uncert["low_lim"]
            img_uncert_ub[:, water_pixels[0], water_pixels[1], :] = uncert["high_lim"]

    # FIXED: Ensure uncertainty datasets are assigned to data_vars outside the water conditional.
    # This prevents Dask schema mismatches on pure-land chunks.
    if return_uncert:
        data_vars["uncertainty_low"] = (("model", "y", "x", "output"), img_uncert_lb.astype(np.float32))
        data_vars["uncertainty_high"] = (("model", "y", "x", "output"), img_uncert_ub.astype(np.float32))

    # PACKAGING PORTION: Return the Dataset natively
    ds_chunk = xr.Dataset(
        data_vars=data_vars,
        coords={
            "model": np.arange(n_models),
            "y": img_chunk_t.y,
            "x": img_chunk_t.x,
            "output": np.arange(n_outputs),
        }
    )
    return ds_chunk


def map_cube_mdn(
    img_data: xr.DataArray,
    sensor: str = "OLCI",
    products: str = "chl",
    wvl_bands: Union[List[float], np.ndarray] = None,
    op_mode: str = "select",
    return_uncert: bool = True,
    uncert_mode: str = "limits",
    scaler_mode: str = "invert",
    land_mask: bool = False,
    landmask_threshold: float = 0.0,
    progress_vis: bool = True,
):
    """
    Map an image cube using MDN to produce predictions and uncertainties.

    Parameters
    ----------
    args : dict or object
        Configuration/argument namespace for the MDN model pipeline.
    img_data : xarray.DataArray
        Image cube (nRow x nCols x nBands).
    sensor : str
        Sensor name (e.g., "OLCI").
    products : str
        Comma-separated products to predict and extract (e.g., "chl,aph").
    wvl_bands : array-like
        Wavelengths corresponding to img_data bands.
    op_mode : {"select", "full"}
        Whether to select the median model or return full ensemble.
    return_uncert : bool
        Whether to return uncertainties.
    uncert_mode : {"composite", "limits"}
        How uncertainties are returned.
    scaler_mode : {"invert", "non_invert"}
        Whether to invert scaled predictions.
    land_mask : bool
        Apply heuristic land masking.
    landmask_threshold : float
        Threshold for land mask.
    progress_vis : bool
        Flag controlling the progress behavior.

    Returns
    -------
    final_ds : xarray.Dataset
        Processed outputs containing predictions and optional uncertainties.
    op_slices : dict
        Slices mapping each product to its respective index in the output dimension.
    """
    # ------------------------
    # Validate inputs
    # ------------------------
    if not isinstance(img_data, xr.DataArray):
        raise TypeError(f"Expected 'img_data' to be an xarray.DataArray, got {type(img_data).__name__}")
    if img_data.ndim != 3:
        raise ValueError(f"Expected 'img_data' to have exactly 3 dimensions (bands, y, x), got ndim={img_data.ndim}")

    if not bool(re.match(PRODUCT_PATTERN, products)):
        raise ValueError(f"Invalid 'products' format: '{products}'. Expected a comma-separated string.")

    if scaler_mode not in ["invert", "non_invert"]:
        raise ValueError(f"Invalid 'scaler_mode': '{scaler_mode}'.")

    if op_mode not in ["select", "full"]:
        raise ValueError(f"Invalid 'op_mode': '{op_mode}'.")

    if uncert_mode not in ["composite", "limits"]:
        raise ValueError(f"Invalid 'uncert_mode': '{uncert_mode}'.")

    if wvl_bands is None:
        wvl_bands = get_sensor_bands(sensor)
    if not isinstance(wvl_bands, (list, np.ndarray)):
        raise TypeError(f"Expected 'wvl_bands' to be a list or numpy.ndarray, got {type(wvl_bands).__name__}")
        
    if len(wvl_bands) != img_data.shape[0]:
        raise ValueError(
            f"Dimension mismatch: length of 'wvl_bands' ({len(wvl_bands)}) must match "
            f"the spectral band axis dimension of 'img_data' ({img_data.shape[0]})"
        )

    # ----------------------------------------------------
    # Safe Conditional Chunking Guard
    # ----------------------------------------------------
    # If the user passed an unchunked array, fallback to spatial chunking.
    # If it is already chunked, this block is completely skipped!
    if img_data.chunks is None:
        img_data = img_data.chunk({"band": -1, "y": 100, "x": 100})

    # Fetch pipeline configurations
    kwargs = get_default_pipeline_kwargs(sensor=sensor, product=products)
    args = get_args(**kwargs)

    # Match model bands to image bands
    sensor_bands = get_sensor_bands(args.sensor)
    valid_bands = []
    for b in sensor_bands:
        idx = np.argmin(np.abs(np.asarray(wvl_bands) - b))
        if np.abs(wvl_bands[idx] - b) > 5:
            raise ValueError(f"Image bands {wvl_bands} do not match sensor bands {sensor_bands}")
        valid_bands.append(idx)

    # Compute expected metadata shapes
    n_rounds = args.get("n_rounds", args["n_rounds"]) if isinstance(args, dict) else args.n_rounds
    n_models = 1 if op_mode == "select" else n_rounds
    n_outputs = len(args.product.split(","))

    # ----------------------------------------------------
    # Generate Dask-backed template arrays
    # ----------------------------------------------------
    # Dynamically match YOUR existing chunks for the spatial dimensions
    y_chunks = img_data.chunks[1] 
    x_chunks = img_data.chunks[2] 
    
    # Establish new dimension chunks
    model_chunks = (n_models,)
    output_chunks = (n_outputs,)
    template_chunks = (model_chunks, y_chunks, x_chunks, output_chunks)

    # Use dask.array.empty to mimic chunk structure safely without loading to memory
    template_vars = {
        "predictions": (
            ("model", "y", "x", "output"), 
            da.empty((n_models, len(img_data.y), len(img_data.x), n_outputs), chunks=template_chunks, dtype=np.float32)
        )
    }
    if return_uncert:
        template_vars["uncertainty_low"] = (
            ("model", "y", "x", "output"), 
            da.empty((n_models, len(img_data.y), len(img_data.x), n_outputs), chunks=template_chunks, dtype=np.float32)
        )
        template_vars["uncertainty_high"] = (
            ("model", "y", "x", "output"), 
            da.empty((n_models, len(img_data.y), len(img_data.x), n_outputs), chunks=template_chunks, dtype=np.float32)
        )

        if uncert_mode == "composite":
            template_vars["uncertainty_comp"] = (
                ("model", "y", "x", "output"), 
                da.empty((n_models, len(img_data.y), len(img_data.x), n_outputs), chunks=template_chunks, dtype=np.float32)
            )

    template_ds = xr.Dataset(
        data_vars=template_vars,
        coords={
            "model": np.arange(n_models),
            "y": img_data.y,
            "x": img_data.x,
            "output": np.arange(n_outputs),
        }
    )

    # Fetch op_slices once globally using a dummy single pixel vector
    dummy_pixel = 0.01 * np.ones((1, len(valid_bands)))
    _, op_slices =  get_spectral_preds(test_x=dummy_pixel, sensor= sensor, products=products, op_mode=op_mode, return_uncert=False, uncert_mode="limits",
                    progress_vis=False)

    # Run mapping across dask-chunk blocks lazily
    lazy_result_ds = xr.map_blocks(
        map_cube_mdn_chunk,
        img_data, 
        kwargs={
            "args": args,
            "target_products": products,
            "wvl_bands": wvl_bands,
            "op_mode": op_mode,
            "land_mask": land_mask,
            "landmask_threshold": landmask_threshold,
            "scaler_mode": scaler_mode,
            "return_uncert": return_uncert
        },
        template=template_ds
    )

    # If composite mode is selected, compute the difference lazily
    if return_uncert and uncert_mode == "composite":
        lazy_result_ds["uncertainty_comp"] = (
            lazy_result_ds["uncertainty_high"] - lazy_result_ds["uncertainty_low"]
        )

    # Compute execution with progress bar
    if progress_vis:
        with ProgressBar():
            final_ds = lazy_result_ds.compute()
    else:
        final_ds = lazy_result_ds.compute()

    # Save metadata dictionary attributes
    #final_ds.attrs["op_slices"] = op_slices

    return final_ds, op_slices