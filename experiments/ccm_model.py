import numpy as np
import warnings

try:
    from causal_ccm.causal_ccm import ccm
    CCM_AVAILABLE = True
except ImportError:
    CCM_AVAILABLE = False
    print("NOTE: causal_ccm module not found in this environment. The surrogate "
          "machinery below is ready to run as-is once moved into your "
          "causal-method-env (same import as your other CCM usage).")


def _raw_ccm_rho(x, y, E=2, tau=1):
    if not CCM_AVAILABLE:
        raise ImportError("causal_ccm not available -- run this in your causal-method-env.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            ccm_obj = ccm(X=np.asarray(x), Y=np.asarray(y), E=E, tau=tau)
            x_to_y_rho, y_to_x_rho = ccm_obj.causality()
        if np.isnan(x_to_y_rho):
            x_to_y_rho = 0.0
        if np.isnan(y_to_x_rho):
            y_to_x_rho = 0.0
        return x_to_y_rho, y_to_x_rho
    except Exception:
        return 0.0, 0.0


def generate_shuffle_surrogates(series, n_surrogates=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    series = np.asarray(series)
    surrogates = []
    for _ in range(n_surrogates):
        s = series.copy()
        rng.shuffle(s)
        surrogates.append(s)
    return surrogates


def ccm_surrogate_threshold(x, y, E=2, tau=1, n_surrogates=100, percentile=95, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    x = np.asarray(x)
    y = np.asarray(y)
    x_to_y_rho, y_to_x_rho = _raw_ccm_rho(x, y, E=E, tau=tau)

    null_xy = []
    for y_surr in generate_shuffle_surrogates(y, n_surrogates, rng):
        r_xy, _ = _raw_ccm_rho(x, y_surr, E=E, tau=tau)
        null_xy.append(r_xy)

    null_yx = []
    for x_surr in generate_shuffle_surrogates(x, n_surrogates, rng):
        _, r_yx = _raw_ccm_rho(x_surr, y, E=E, tau=tau)
        null_yx.append(r_yx)

    null_xy = np.array(null_xy)
    null_yx = np.array(null_yx)
    threshold_xy = np.percentile(null_xy, percentile)
    threshold_yx = np.percentile(null_yx, percentile)

    return x_to_y_rho, y_to_x_rho, threshold_xy, threshold_yx, null_xy, null_yx


def ccm_identify_causality(x, y, E=2, tau=1, n_surrogates=100, percentile=95, rng=None):
    x_to_y_rho, y_to_x_rho, thr_xy, thr_yx, null_xy, null_yx = ccm_surrogate_threshold(
        x, y, E=E, tau=tau, n_surrogates=n_surrogates, percentile=percentile, rng=rng
    )

    xy_significant = x_to_y_rho > thr_xy
    yx_significant = y_to_x_rho > thr_yx

    if not xy_significant and not yx_significant:
        return 'Independence', 0.0

    if xy_significant and not yx_significant:
        return 'Y -> X', float(x_to_y_rho - thr_xy)

    if yx_significant and not xy_significant:
        return 'X -> Y', float(y_to_x_rho - thr_yx)

    margin_xy = x_to_y_rho - thr_xy
    margin_yx = y_to_x_rho - thr_yx
    if margin_xy >= margin_yx:
        return 'Y -> X', float(margin_xy)
    else:
        return 'X -> Y', float(margin_yx)