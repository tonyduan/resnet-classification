import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from matplotlib import pyplot as plt


def temperature_scale(labels, logits):
    """
    Run post-hoc temperature scaling [Guo et al. 2017].

    Parameters
    ----------
    labels: (m, n) array of one-hot labels
    logits: (m, n) array of predicted logits

    Returns
    -------
    T: scalar float for optimal temperature
    """
    T = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.0001, max_iter=2000, tolerance_grad=1e-3,
                                  line_search_fn="strong_wolfe")
    def closure():
        loss = F.cross_entropy(logits / T, labels, reduction="mean")
        loss.backward()
        return loss
    optimizer.step(closure)
    T = np.asscalar(T.data.numpy())
    return T


def calibration_curve(labels, preds, n_bins=15, eps=1e-8, raise_on_nan=True):
    """
    Returns calibration curve at the pre-specified number of bins.

    Parameters
    ----------
    labels: (m,)-length array of binary labels in {0, 1}
    preds: (m,)-length array of binary predictions in [0, 1]
    n_bins: integer, number of bins over [0,1] (discretized uniformly)
    eps: float, for numerical stability
    raise_on_nan: boolean, if True will raise error if a bin is empty
                           if False returns without error and presumably weighted to zero by a
                           downstream averaging procedure like in calibration_error

    Returns
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

    Parameters
    ----------
    obs_cdfs: (n_bins,)-length array of observed cdfs in each bin
    pred_cdfs: (n_bins,)-length array of predicted cdfs in each bin
    bin_cnts: (n_bins,)-length array of counts, for averaging
    p: either 1 or 2 for L1 or L2 calibration error, note L1 corresponds to well-known ECE
    n_mc_samples: integer, used for bootstrap de-biasing of L1 calibration error

    Returns
    -------
    calibration_error: de-biased L-p calibration error
    """
    if p == 2:
        cnts_clip = np.clip(bin_cnts, a_min=2, a_max=None)  # a bit hacky, but prevents nans
        per_bin_calib = (obs_cdfs - pred_cdfs) ** 2 - obs_cdfs * (1 - obs_cdfs) / (cnts_clip - 1)
        return np.average(per_bin_calib, weights=bin_cnts) ** 0.5
    elif p == 1:
        plugin_calib = np.average(np.abs(obs_cdfs - pred_cdfs), weights=bin_cnts)
        cnts_clip = np.clip(bin_cnts, a_min=1, a_max=None)
        mc_samples = np.random.randn(n_mc_samples, len(bin_cnts))
        mc_samples = mc_samples * (obs_cdfs * (1 - obs_cdfs) / cnts_clip) ** 0.5 + obs_cdfs
        mc_calib = np.mean(np.average(np.abs(pred_cdfs - mc_samples), axis=1, weights=bin_cnts))
        return plugin_calib - (mc_calib - plugin_calib)
    raise ValueError


def plot_calibration_curve(obs_cdfs, pred_cdfs):
    plt.figure(figsize=(5, 5))
    plt.scatter(pred_cdfs, obs_cdfs, color="black")
    plt.plot((0, 1), (0, 1), "--", color="grey")
    plt.xlim((0, 1))
    plt.xlabel("Expected CDF")
    plt.ylim((0, 1))
    plt.ylabel("Observed CDF")
