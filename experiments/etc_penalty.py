# ETC-Penalty (ETCP) causal direction wrapper
# Mirrors the LZP implementation: raw complexity values are pulled from
# ETCPy, then causal direction is decided via a surrogate-based permutation
# test on the gap between the two directional values.

import numpy as np
import ETCPy.ETC.CCMC.pairs as pairs

# TODO: confirm this is the actual function name in your ETCPy version.
# This assumes ETCP_causality exists with the same return shape as
# LZ_causality (a dict with '..._x_to_y' / '..._y_to_x' keys). If it
# doesn't exist, run `dir(pairs)` to see what's available, or fall back to
# CCM_causality's 'ETCP_direction' key as used in your earlier script.

def compute_etcp_values(x, y):
    try:
        etcp_dict = pairs.ETC_causality(x, y)
        etcp_xy = etcp_dict['ETCP_x_to_y']
        etcp_yx = etcp_dict['ETCP_y_to_x']

        return etcp_xy, etcp_yx

    except Exception as e:
        print('ETCP error:', e)
        return np.nan, np.nan


def etcp_surrogate_threshold(X, Y, n_surrogates=100, percentile=95):
    """
    Estimate the gap distribution under the null hypothesis of independence
    by randomly shuffling X and Y and measuring the gap.
    """
    gaps = []
    # Convert to list of characters/elements for shuffling
    X_list = list(X)
    Y_list = list(Y)

    for _ in range(n_surrogates):
        # Shuffle both to destroy any causal structure
        X_shuffled = np.random.permutation(X_list)
        Y_shuffled = np.random.permutation(Y_list)

        # Recompute stats for shuffled pair
        etcp_xy, etcp_yx = compute_etcp_values(X_shuffled, Y_shuffled)
        gap = abs(etcp_xy - etcp_yx)
        if not np.isnan(gap):
            gaps.append(gap)

    if not gaps:
        # every surrogate call failed (e.g. underlying function name is wrong) --
        # returning inf makes the real observed gap fail the significance test
        # below instead of silently passing through it as 'Y -> X'
        return np.inf

    return np.percentile(gaps, percentile)


# ETCP identify causality
def etcp_identify_causality(x, y):
    etcp_xy, etcp_yx = compute_etcp_values(x, y)

    # FIX: guard against NaN from a broken underlying call. Without this,
    # nan comparisons are always False in Python, so a broken pairs.ETC_causality
    # call would silently fall through to 'Y -> X' with gap=nan every single
    # time instead of surfacing the failure or returning 'Independence'.
    if np.isnan(etcp_xy) or np.isnan(etcp_yx):
        return 'Independence', 0.0

    gap = etcp_xy - etcp_yx
    abs_gap = abs(gap)

    threshold = etcp_surrogate_threshold(x, y, n_surrogates=100)

    if abs_gap < threshold:
        return 'Independence', abs_gap
    if gap < 0:
        return 'X -> Y', abs_gap
    else:
        return 'Y -> X', abs_gap