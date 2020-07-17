import numpy as np


def calibration_curve(labels, preds, n_bins=10, eps=1e-8, raise_on_nan=True):
    """
    Returns calibration curve at the pre-specified number of bins.
    -------
    obs_cdfs: (n_bins,)
    pred_cdfs: (n_bins,)
    bin_cnts: (n_bins,)
    """
    bins = np.linspace(0., 1. + eps, n_bins + 1)
    bin_ids = np.digitize(preds, bins) - 1
    bin_cnts = np.bincount(bin_ids, minlength=n_bins)
    pred_cdfs = np.bincount(bin_ids, weights=preds, minlength=n_bins)
    obs_cdfs = np.bincount(bin_ids, weights=labels, minlength=n_bins)

    if np.any(bin_cnts == 0) and raise_on_nan:
        raise ValueError("Exists a bin with no predictions. Reduce the number of bins.")
    else:
        pred_cdfs = pred_cdfs / np.clip(bin_cnts, a_min=1, a_max=None)
        obs_cdfs = obs_cdfs / np.clip(bin_cnts, a_min=1, a_max=None)

    return obs_cdfs, pred_cdfs, bin_cnts


def calibration_error(obs_cdfs, pred_cdfs, bin_cnts, p=2, n_mc_samples=1000):
    """
    De-biased L-p calibration error [Kumar et al. NeurIPS 2019], where p in {1, 2}.

    Todo
    ----
    Add confidence intervals via bootstrap re-sampling.
    """
    if p == 2:
        bin_cnts = np.clip(bin_cnts, a_min=2, a_max=None)
        per_bin_calib = (obs_cdfs - pred_cdfs) ** 2 - obs_cdfs * (1 - obs_cdfs) / (bin_cnts - 1)
        return np.average(per_bin_calib, weights=bin_cnts) ** 0.5
    elif p == 1:
        plugin_calib = np.average(np.abs(obs_cdfs - pred_cdfs), weights=bin_cnts)
        obs_cdfs = obs_cdfs[:, np.newaxis]
        mc_samples = np.random.randn(len(bin_cnts), n_mc_samples)
        bin_cnts = np.clip(bin_cnts, a_min=1, a_max=None)
        mc_samples = obs_cdfs * (1 - obs_cdfs) / bin_cnts[:, np.newaxis] * mc_samples + obs_cdfs
        mc_calib = np.mean(np.average(np.abs(obs_cdfs - mc_samples), axis=0, weights=bin_cnts))
        return plugin_calib - (mc_calib - plugin_calib)
    raise ValueError

