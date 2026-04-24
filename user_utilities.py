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

from typing import Optional, Tuple
import numpy as np

from .meta import get_sensor_bands
from .parameters import get_args
from .utilities import get_mdn_preds_uncertainties, get_mdn_preds_raw

#rgb_bands = [660, 550, 440]
min_in_out_val = 1e-6

# If new default model is defined for a sensor this Dictionary needs to be updated.
DEFAULT_SENSOR_PRODUCT_COMBINATIONS = {
    "OLCI": 'chl,tss,cdom',
    "PACE-delivery": 'aph,chl,tss,pc,ad,ag,cdom',
    "SD8-cc_base": 'chl,secchi',
}

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

    outputs, op_slices = get_mdn_preds_raw(test_x, args=args, op_mode="full",
                                                                 scaler_mode="non_invert")

    return outputs, op_slices


def get_spectral_preds(
        test_x: np.ndarray,
        sensor: str = "OLCI",
        products: str = "chl",
        # op_mode: str = "select",
        return_uncert:bool = "True",
        uncert_mode: str = "limits",
        scaler_mode: str = "invert",
        progress_vis: bool= True
    ) -> Tuple[np.ndarray, dict]:
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

    return_uncert : bool
        Whether the function returns the uncertainties. (Default: True)

    uncert_mode : {"composite", "limit"}
        Defines the mode in which the uncertainty is returned:
            - "limit": returns the upper and lower limits as estimated from the predicted distribution
            - "composite": returns the average distance on each side

    scaler_mode : {"invert", "non_invert"}
        Whether to apply inverse scaling to outputs. (Default: invert)

    progress_vis : bool
        Whether the progress of the tqdms are shown on screen. (Default: True)

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

    # Now call the existing predictor function with these arguments
    mdn_preds, mdn_uncert, mdn_preds_slices = get_mdn_preds_uncertainties(test_x=test_x, args=args, op_mode="select", 
                                                    scaler_mode=scaler_mode, uncert_mode=uncert_mode, progress_vis=progress_vis)

    # Now only extract the output corresponding to the needed products
    products = [s.strip() for s in products.split(',')]
    selected_slices = [mdn_preds_slices[key] for key in products]
    
    # If multiple products are selected we need to select the outputs and uncertainties across the products. If only one product is selected we can just return the output as is without concatenation
    if len(args.product.split(',')) != 1:
        estimates, uncert = {key: mdn_preds[key][:, np.r_[tuple(selected_slices)]] for key in mdn_preds.keys()}, {key: mdn_uncert[key][:, np.r_[tuple(selected_slices)]] for key in mdn_uncert.keys()}
    else:
        estimates, uncert = mdn_preds, mdn_uncert
    
    if return_uncert:
        return estimates, uncert, selected_slices
    else:
        return estimates, selected_slices



                





