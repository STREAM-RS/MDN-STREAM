# -*- coding: utf-8 -*-
"""
File Name:      utilities
Description:    This code file contains a set of supporting functions that are needed to generate the predictions and
                uncertainty estimations available for a specific models.

Date Created:   September 2nd, 2024
"""

from typing import Optional, Tuple
import numpy as np
import numpy as np
from tqdm import tqdm


from .meta import get_sensor_bands
from .parameters import get_args
from .product_estimation import get_estimates
from .uncertainty_package_final.uncert_support_lib import get_sample_uncertainity
from .utils import mask_land

'Base properties for imshow'
ASPECT = 'auto'
cmap = 'jet'


def get_mdn_preds_raw(
    test_x: np.ndarray,
    args: Optional[dict] = None,
    sensor: str = "OLCI",
    products: str = "chl",
    op_mode: str = "select",
    scaler_mode: str = "invert",
    model_type: str = "production",
    model_uid: Optional[str] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Directly outputs the predictions from the MDN with no processing

    Parameters
    ----------
    test_x : np.ndarray
        Input data (n_samples x n_features).

    args : dict, optional
        MDN arguments. If None, defaults are generated.

    sensor : str
        Sensor name (used if args is None).

    products : str
        Products to predict (used if args is None).

    op_mode : {"point", "full"}
        - "point": return only max-likelihood predictions.
        - "full": return full MDN output suite.

    scaler_mode : {"invert", "non_invert"}
        Whether to apply inverse scaling to outputs.

    model_type : {"production", "testing"}
        Which MDN model type to use.

    model_uid : str, optional
        Specific model UID to load.

    verbose : bool
        Print debug info if True.

    Returns
    -------
    output : np.ndarray
        Predictions (shape depends on mode):
        - "point": (n_samples, n_outputs)
        - "full": (n_models, n_samples, n_outputs)

    op_slices : dict
        Dictionary of output slices per predicted product.
    """

    # ----------------------------------------
    # Validate arguments
    # ----------------------------------------
    if model_type not in ["production", "testing"]:
        raise ValueError(f"model_type must be 'production' or 'testing'. Got '{model_type}'.")
    if op_mode not in ["select", "full"]:
        raise ValueError(f"mode must be 'select' or 'full'. Got '{op_mode}'.")
    if scaler_mode not in ["invert", "non_invert"]:
        raise ValueError(f"scaler_mode must be 'invert' or 'non_invert'. Got '{scaler_mode}'.")

    # ----------------------------------------
    # Generate default args if needed
    # ----------------------------------------
    if args is None:
        if verbose:
            print(f"No MDN args provided. Generating defaults for sensor={sensor}, product={products}")

        kwargs = {
            'product': products,
            'sensor': sensor,
            'model_loc': "Weights" if model_type == "production" else "Weights_test",
            'sat_bands': True if products in ['chl','tss','cdom','pc'] else False,
        }

        if model_uid is not None:
            kwargs['model_uid'] = model_uid

        args = get_args(kwargs, use_cmdline=False)

    elif verbose:
        print(f"Using provided MDN args for sensor={args.sensor}, product={args.products}")

    # ----------------------------------------
    # Get MDN predictions
    # ----------------------------------------
    if test_x is not None:
        outputs, op_slices = get_estimates(args, x_test=test_x, return_coefs=True)
        _ = outputs.pop('estimates')

    else:
        raise ValueError("test_x cannot be None")

    return outputs, op_slices
def get_mdn_preds(
    test_x: np.ndarray,
    args: Optional[dict] = None,
    sensor: str = "OLCI",
    products: str = "chl",
    op_mode: str = "select",
    scaler_mode: str = "invert",
    model_type: str = "production",
    model_uid: Optional[str] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Generate MDN predictions for a given input dataset.

    Parameters
    ----------
    test_x : np.ndarray
        Input data (n_samples x n_features).

    args : dict, optional
        MDN arguments. If None, defaults are generated.

    sensor : str
        Sensor name (used if args is None).

    products : str
        Products to predict (used if args is None).

    op_mode : {"point", "full"}
        - "point": return only max-likelihood predictions.
        - "full": return full MDN output suite.

    scaler_mode : {"invert", "non_invert"}
        Whether to apply inverse scaling to outputs.

    model_type : {"production", "testing"}
        Which MDN model type to use.

    model_uid : str, optional
        Specific model UID to load.

    verbose : bool
        Print debug info if True.

    Returns
    -------
    output : np.ndarray
        Predictions (shape depends on mode):
        - "point": (n_samples, n_outputs)
        - "full": (n_models, n_samples, n_outputs)

    op_slices : dict
        Dictionary of output slices per predicted product.
    """

    # ----------------------------------------
    # Validate arguments
    # ----------------------------------------
    if model_type not in ["production", "testing"]:
        raise ValueError(f"model_type must be 'production' or 'testing'. Got '{model_type}'.")
    if op_mode not in ["select", "full"]:
        raise ValueError(f"mode must be 'select' or 'full'. Got '{op_mode}'.")
    if scaler_mode not in ["invert", "non_invert"]:
        raise ValueError(f"scaler_mode must be 'invert' or 'non_invert'. Got '{scaler_mode}'.")

    # ----------------------------------------
    # Generate default args if needed
    # ----------------------------------------
    if args is None:
        if verbose:
            print(f"No MDN args provided. Generating defaults for sensor={sensor}, product={products}")

        kwargs = {
            'product': products,
            'sensor': sensor,
            'model_loc': "Weights" if model_type == "production" else "Weights_test",
            'sat_bands': True if products in ['chl','tss','cdom','pc'] else False,
        }

        if model_uid is not None:
            kwargs['model_uid'] = model_uid

        args = get_args(kwargs, use_cmdline=False)

    elif verbose:
        print(f"Using provided MDN args for sensor={args.sensor}, product={args.products}")

    # ----------------------------------------
    # Get MDN predictions
    # ----------------------------------------
    if test_x is not None:
        outputs, op_slices = get_estimates(args, x_test=test_x, return_coefs=True)

        # Extract MLE predictions from MDN ensemble
        output = get_mdn_mle_matrix(
            mdn_outputs=outputs['coefs'],
            scalers=outputs['scalery'],
            scaler_mode=scaler_mode,
            op_mode=op_mode
        )

    else:
        raise ValueError("test_x cannot be None")

    return output, op_slices


def get_mdn_mle_matrix(
    mdn_outputs,
    scalers=None,
    scaler_mode="invert",
    op_mode="full"
):
    """
    Compute MLE predictions from an ensemble of MDNs, returning a clean matrix.

    Parameters
    ----------
    mdn_outputs : list
        Length = n_models.
        Each item is a dict-like structure with:
            item[0] = weights    (n_samples, n_components)
            item[1] = means      (n_samples, n_components, n_outputs)
            item[2] = variances  (unused)

    scalers : list or None
        Required if scaler_mode == 'invert'. Must be length n_models.

    scaler_mode : {"invert", "non_invert"}
        - "invert"     => return only inverse-transformed values
        - "non_invert" => return only scaled values

    op_mode : {"full", "select"}
        - "full"   => return predictions of all models (3-D array)
        - "select" => return predictions of the model closest to ensemble median (2-D array, sample-wise)

    Returns
    -------
    output : ndarray
        Predictions:
        - full: (n_models, n_samples, n_outputs)
        - select: (n_samples, n_outputs)

    selected_index : ndarray or None
        For "select": array of shape (n_samples,) indicating which model was chosen per sample
        For "full": None
    """

    # ----------------------------------------
    # Validate modes
    # ----------------------------------------
    if scaler_mode not in ["invert", "non_invert"]:
        raise ValueError("scaler_mode must be 'invert' or 'non_invert'.")
    if op_mode not in ["full", "select"]:
        raise ValueError("op_mode must be 'full' or 'select'.")

    n_models = len(mdn_outputs)

    # ----------------------------------------
    # Check scalers if needed
    # ----------------------------------------
    if scaler_mode == "invert":
        if scalers is None:
            raise ValueError("scalers must be provided when scaler_mode='invert'.")
        if len(scalers) != n_models:
            raise ValueError("Number of scalers must match number of MDNs.")

    # Get n_samples, n_outputs from first MDN
    n_samples = mdn_outputs[0][0].shape[0]
    n_outputs = mdn_outputs[0][1].shape[2]

    # ----------------------------------------
    # Compute MLE in scaled space for each MDN
    # ----------------------------------------
    mle_scaled = np.zeros((n_models, n_samples, n_outputs))

    for i in tqdm(range(n_models), desc="Extracting MDN MLE"):
        weights = mdn_outputs[i][0]
        means   = mdn_outputs[i][1]
        max_idx = np.argmax(weights, axis=1)

        for s in range(n_samples):
            mle_scaled[i, s] = means[s, max_idx[s]]

    # ----------------------------------------
    # Selection mode (per sample, L1 distance)
    # ----------------------------------------
    selected_index = None
    if op_mode == "select":
        # Transpose to (n_samples, n_models, n_outputs) for per-sample distances
        mle_t = np.transpose(mle_scaled, (1,0,2))  # (n_samples, n_models, n_outputs)
        # Median across models for each sample
        median_pred = np.median(mle_t, axis=1, keepdims=True)  # (n_samples, 1, n_outputs)
        # Compute L1 distance per sample and model
        distances = np.sum(np.abs(mle_t - median_pred), axis=2)  # L1 distance
        # Select model closest to median for each sample
        selected_index = np.argmin(distances, axis=1)  # (n_samples,)

        # Gather predictions per sample
        output = np.array([mle_t[s, selected_index[s]] for s in range(n_samples)])  # (n_samples, n_outputs)

    else:
        # full predictions
        output = mle_scaled
        selected_index = None

    # ----------------------------------------
    # Apply inverse scaling if requested
    # ----------------------------------------
    if scaler_mode == "invert":
        if op_mode == "full":
            inv_output = np.zeros_like(output)
            for i in range(n_models):
                inv_output[i] = scalers[i].inverse_transform(output[i])
            output = inv_output
        else:  # select
            for s in range(n_samples):
                output[s] = scalers[selected_index[s]].inverse_transform(output[s].reshape(1, -1))[0]

    return {
        "output": output,                # matrix ONLY in one space (scaled OR inverted)
        "selected_index": selected_index
    }


def get_mdn_preds_uncertainties(
    test_x: np.ndarray,
    args: Optional[dict] = None,
    sensor: str = "OLCI",
    products: str = "chl",
    op_mode: str = "select",
    scaler_mode: str = "invert",
    uncert_mode: str = "composite",
    model_type: str = "production",
    model_uid: Optional[str] = None,
    verbose: bool = False
) -> Tuple[dict, dict, dict]:
    """
    Generate MDN predictions and uncertainties for a given input dataset.

    Parameters
    ----------
    test_x : np.ndarray
        Input data (n_samples x n_features).

    args : dict, optional
        MDN arguments. If None, defaults are generated.

    sensor : str
        Sensor name (used if args is None).

    products : str
        Products to predict (used if args is None).

    op_mode : {"select", "full"}
        - "select": return only max-likelihood predictions closest to ensemble median.
        - "full": return predictions for all models.

    scaler_mode : {"invert", "non_invert"}
        Whether to apply inverse scaling to outputs.

    uncert_mode: {"composite", "limits"}
        Whether uncertainties are returned as composite SD or as limits.

    model_type : {"production", "testing"}
        Which MDN model type to use.

    model_uid : str, optional
        Specific model UID to load.

    verbose : bool
        Print debug info if True.

    Returns
    -------
    predictions_dict : dict
        'pred' : ndarray
            - "select": (n_samples, n_outputs)
            - "full": (n_models, n_samples, n_outputs)
        'selected_index' : None for "full", or array (n_samples,) for "select"

    uncertainties : dict
        - composite mode: {'comp_unc': ...}
        - limits mode: {'low_lim': ..., 'high_lim': ...}

    op_slices : dict
        Dictionary of output slices per predicted product.
    """

    # ------------------------
    # Validate arguments
    # ------------------------
    if model_type not in ["production", "testing"]:
        raise ValueError(f"model_type must be 'production' or 'testing'. Got '{model_type}'")
    if op_mode not in ["select", "full"]:
        raise ValueError(f"mode must be 'select' or 'full'. Got '{op_mode}'")
    if scaler_mode not in ["invert", "non_invert"]:
        raise ValueError(f"scaler_mode must be 'invert' or 'non_invert'. Got '{scaler_mode}'")
    if uncert_mode not in ["composite", "limits"]:
        raise ValueError(f"uncert_mode must be 'composite' or 'limits'. Got '{uncert_mode}'")

    # ------------------------
    # Generate default args if needed
    # ------------------------
    if args is None:
        if verbose:
            print(f"No MDN args provided. Generating defaults for sensor={sensor}, product={products}")
        kwargs = {
            'product': products,
            'sensor': sensor,
            'model_loc': "Weights" if model_type == "production" else "Weights_test",
            'sat_bands': products in ['chl', 'tss', 'cdom', 'pc'],
        }
        if model_uid is not None:
            kwargs['model_uid'] = model_uid
        args = get_args(kwargs, use_cmdline=False)
    elif verbose:
        print(f"Using provided MDN args for sensor={args.sensor}, product={args.products}")

    # ------------------------
    # Get MDN predictions
    # ------------------------
    if test_x is None:
        raise ValueError("test_x cannot be None")

    outputs, op_slices = get_estimates(args, x_test=test_x, return_coefs=True)

    predictions_dict, uncertainties = get_mdn_predictions_and_uncertainties(
        mdn_outputs=outputs['coefs'],
        scalers=outputs['scalery'],
        scaler_mode=scaler_mode,
        op_mode=op_mode,
        uncert_mode=uncert_mode
    )

    return predictions_dict, uncertainties, op_slices


def get_mdn_predictions_and_uncertainties(
    mdn_outputs,
    scalers=None,
    scaler_mode="non_invert",
    op_mode="full",
    uncert_mode="composite"
):
    """
    Compute MDN predictions (MLE) and uncertainties (aleatoric + epistemic) for an ensemble of MDNs.

    Parameters
    ----------
    mdn_outputs : list
        Each element is a tuple/list:
            item[0] = pred_wts    (n_samples, n_components)
            item[1] = pred_mu     (n_samples, n_components, n_outputs)
            item[2] = pred_sigma  (n_samples, n_components, n_outputs, n_outputs)

    scalers : list or None
        Required if scaler_mode == "invert". Length must match number of models.

    scaler_mode : {"invert", "non_invert"}
        - "invert": return inverse-transformed values using provided scalers
        - "non_invert": return scaled values

    op_mode : {"full", "select"}
        - "full": return predictions and uncertainties for all models
        - "select": select model closest to ensemble median per sample

    uncert_mode : {"composite", "limits"}
        - "composite": return SD as 'comp_unc'
        - "limits": return (low, high) bounds

    Returns
    -------
    predictions : dict
        'pred' : ndarray
            - full: (n_models, n_samples, n_outputs)
            - select: (n_samples, n_outputs)
        'selected_index' : None for full, or array (n_samples,) for select

    uncertainties : dict
        - composite mode: {'comp_unc': SD}
        - limits mode: {'low_lim': low, 'high_lim': high}
    """

    # ------------------------
    # Input validation
    # ------------------------
    if scaler_mode not in ["invert", "non_invert"]:
        raise ValueError("scaler_mode must be either 'invert' or 'non_invert'.")
    if op_mode not in ["full", "select"]:
        raise ValueError("op_mode must be either 'full' or 'select'.")
    if uncert_mode not in ["composite", "limits"]:
        raise ValueError("uncert_mode must be either 'composite' or 'limits'.")

    n_models = len(mdn_outputs)
    n_samples = mdn_outputs[0][0].shape[0]
    n_outputs = mdn_outputs[0][1].shape[2]

    if scaler_mode == "invert":
        if scalers is None or len(scalers) != n_models:
            raise ValueError("scalers must be provided and match number of MDNs when scaler_mode='invert'.")

    # Initialize matrices
    mle_scaled = np.zeros((n_models, n_samples, n_outputs))
    ensemble_uncertainties = np.zeros((n_models, n_samples, n_outputs))

    for m_idx, item in enumerate(tqdm(mdn_outputs, desc="Processing MDN ensemble for uncertainty")):
        pred_wts, pred_mu, pred_sigma = item[0], item[1], item[2]

        # --- Compute MLE ---
        max_idx = np.argmax(pred_wts, axis=1)
        for s in range(n_samples):
            mle_scaled[m_idx, s] = pred_mu[s, max_idx[s]]

        # --- Compute uncertainties ---
        aleatoric, epistemic = get_sample_uncertainity({
            "pred_wts": pred_wts, "pred_mu": pred_mu, "pred_sigma": pred_sigma
        })

        def collapse_unc(u):
            u = np.asarray(u)
            if u.ndim == 2:
                if n_outputs == 1 and u.shape[1] > 1:
                    return np.sum(u, axis=1)[:, None]
                elif u.shape[1] == n_outputs:
                    return u
                else:
                    raise ValueError(f"Unexpected 2D uncertainty shape {u.shape}")
            elif u.ndim == 3:
                return np.sum(u, axis=1)
            else:
                raise ValueError(f"Unexpected uncertainty ndim {u.ndim}")

        ale = collapse_unc(aleatoric)
        epi = collapse_unc(epistemic)
        ensemble_uncertainties[m_idx] = np.sqrt(np.maximum(ale + epi, 0.0))

    # ------------------------
    # Determine selected indices if op_mode is "select"
    # ------------------------
    if op_mode == "select":
        mle_t = np.transpose(mle_scaled, (1, 0, 2))  # (n_samples, n_models, n_outputs)
        median_pred = np.median(mle_t, axis=1, keepdims=True)
        distances = np.sum(np.abs(mle_t - median_pred), axis=2)  # L1 distance
        selected_index = np.argmin(distances, axis=1)  # (n_samples,)

        pred_array = np.array([mle_t[s, selected_index[s]] for s in range(n_samples)])
        uncert_array = np.array([ensemble_uncertainties[selected_index[s], s] for s in range(n_samples)])
    else:
        pred_array = mle_scaled
        uncert_array = ensemble_uncertainties
        selected_index = None

    # ------------------------
    # Compute low/high bounds
    # ------------------------
    low = pred_array - uncert_array
    high = pred_array + uncert_array

    # ------------------------
    # Apply inverse scaling if requested
    # ------------------------
    if scaler_mode == "invert":
        if op_mode == "full":
            for m_idx in range(n_models):
                pred_array[m_idx] = scalers[m_idx].inverse_transform(pred_array[m_idx])
                low[m_idx] = scalers[m_idx].inverse_transform(low[m_idx])
                high[m_idx] = scalers[m_idx].inverse_transform(high[m_idx])
        else:
            for s in range(n_samples):
                pred_array[s] = scalers[selected_index[s]].inverse_transform(pred_array[s].reshape(1, -1))[0]
                low[s] = scalers[selected_index[s]].inverse_transform(low[s].reshape(1, -1))[0]
                high[s] = scalers[selected_index[s]].inverse_transform(high[s].reshape(1, -1))[0]

    # ------------------------
    # Prepare outputs
    # ------------------------
    predictions = {"pred": pred_array, "selected_index": selected_index}

    uncertainties = {"comp_unc": high - low} if uncert_mode == "composite" else {"low_lim": low, "high_lim": high}

    return predictions, uncertainties


import numpy as np
from tqdm import tqdm

def map_cube_mdn_full(
    args,
    img_data: np.ndarray,
    wvl_bands,
    land_mask: bool = False,
    landmask_threshold: float = 0.0,
    flg_subsmpl: bool = False,
    subsmpl_rate: int = 10,
    scaler_mode: str = "invert",
    block_size: int = 10000,
    op_mode: str = "select",
    uncert_mode: str = "composite",
):
    """
    Map an image cube using MDN to produce predictions and uncertainties.

    Parameters
    ----------
    args : dict
        MDN model settings.
    img_data : np.ndarray
        Image cube (nRow x nCols x nBands).
    wvl_bands : array-like
        Wavelengths corresponding to img_data bands.
    land_mask : bool
        Apply heuristic land masking.
    landmask_threshold : float
        Threshold for land mask.
    flg_subsmpl : bool
        Subsample the image.
    subsmpl_rate : int
        Subsampling factor.
    scaler_mode : {"invert", "non_invert"}
        Whether to invert scaled predictions.
    block_size : int
        Number of spectra to process per block.
    op_mode : {"select", "full"}
        Whether to select the median model or return full ensemble.
    uncert_mode : {"composite", "limits"}
        How uncertainties are returned.

    Returns
    -------
    img_preds : np.ndarray
        Prediction cube (nRow x nCols x nOutputs).
    img_uncert : np.ndarray or tuple
        Uncertainty cube or tuple of lower/upper limits.
    op_slices : dict
        Slices for predicted products.
    """

    # ------------------------
    # Validate inputs
    # ------------------------
    assert isinstance(img_data, np.ndarray) and img_data.ndim == 3
    assert isinstance(land_mask, bool)
    assert isinstance(flg_subsmpl, bool)
    assert scaler_mode in ["invert", "non_invert"]
    assert op_mode in ["select", "full"]
    assert uncert_mode in ["composite", "limits"]

    # ------------------------
    # Match model bands to image bands
    # ------------------------
    sensor_bands = get_sensor_bands(args.sensor)
    valid_bands = []
    for b in sensor_bands:
        idx = np.argmin(np.abs(np.asarray(wvl_bands) - b))
        if np.abs(wvl_bands[idx] - b) > 5:
            raise ValueError(f"Image bands {wvl_bands} do not match sensor bands {sensor_bands}")
        valid_bands.append(idx)

    img_data = img_data[:, :, valid_bands]
    wvl_bands = np.asarray(wvl_bands)[valid_bands]

    # ------------------------
    # Subsample if requested
    # ------------------------
    if flg_subsmpl:
        img_data = img_data[::subsmpl_rate, ::subsmpl_rate, :]

    # ------------------------
    # Create water mask
    # ------------------------
    if land_mask:
        img_mask = mask_land(img_data, wvl_bands, threshold=landmask_threshold)
    else:
        img_mask = np.isnan(np.min(img_data, axis=2)).astype(float)

    water_pixels = np.where(img_mask == 0)
    water_spectra = img_data[water_pixels[0], water_pixels[1], :]

    # Remove spectra with majority negative values
    maj_neg = (water_spectra < 1e-4).sum(axis=1) > 5
    water_pixels = tuple(p[~maj_neg] for p in water_pixels)
    water_spectra = water_spectra[~maj_neg]

    if water_spectra.size == 0:
        n_outputs = args['data_ytrain_shape'][1]
        img_preds = np.zeros((img_data.shape[0], img_data.shape[1], n_outputs))
        if uncert_mode == "limits":
            img_uncert_lb = np.zeros_like(img_preds)
            img_uncert_ub = np.zeros_like(img_preds)
            return img_preds, (img_uncert_lb, img_uncert_ub), None
        else:
            img_uncert = np.zeros_like(img_preds)
            return img_preds, img_uncert, None

    # ------------------------
    # Prepare spectra
    # ------------------------
    water_final = np.ma.masked_invalid(water_spectra).reshape((-1, water_spectra.shape[-1]))
    valid_mask = ~np.any(water_final.mask, axis=1)
    water_final = water_final[valid_mask]
    water_pixels = tuple(p[valid_mask] for p in water_pixels)

    # ------------------------
    # Process in blocks
    # ------------------------
    final_estimates = []
    final_uncert = []

    for start in tqdm(range(0, water_final.shape[0], block_size), desc="Processing blocks"):
        block = water_final[start:start + block_size]
        block[block <= args.min_in_out_val] = args.min_in_out_val

        preds, uncert, op_slices = get_mdn_preds_uncertainties(
            block,
            args=args,
            sensor=args.sensor,
            products=args.product,
            scaler_mode=scaler_mode,
            op_mode=op_mode,
            uncert_mode=uncert_mode,
            verbose=False
        )

        final_estimates.append(preds['pred'])
        if uncert_mode == "limits":
            final_uncert.append((uncert['low_lim'], uncert['high_lim']))
        else:
            final_uncert.append(uncert['comp_unc'])

    # Concatenate all blocks
    final_estimates = np.vstack(final_estimates)
    if uncert_mode == "limits":
        low_lim = np.vstack([x[0] for x in final_uncert])
        high_lim = np.vstack([x[1] for x in final_uncert])
    else:
        final_uncert = np.vstack(final_uncert)

    # ------------------------
    # Reconstruct image cubes
    # ------------------------
    n_outputs = final_estimates.shape[1]
    img_preds = np.zeros((img_data.shape[0], img_data.shape[1], n_outputs))
    img_preds[water_pixels[0], water_pixels[1], :] = final_estimates

    if uncert_mode == "limits":
        img_uncert_lb = np.zeros_like(img_preds)
        img_uncert_ub = np.zeros_like(img_preds)
        img_uncert_lb[water_pixels[0], water_pixels[1], :] = low_lim
        img_uncert_ub[water_pixels[0], water_pixels[1], :] = high_lim
        return img_preds, (img_uncert_lb, img_uncert_ub), op_slices
    else:
        img_uncert = np.zeros_like(img_preds)
        img_uncert[water_pixels[0], water_pixels[1], :] = final_uncert
        return img_preds, img_uncert, op_slices

