import numpy as np


def bootstrap_resample(labels):
    """
    Stratified bootstrap re-sample a dataset.

    Parameters
    ----------
    labels: (m,)-length array of integers in {0, ..., num_labels - 1}, where m is number of data pts

    Returns
    -------
    idxs: (m,)-length array of integers in {0, ..., m - 1} representing a resample of indices
    """
    idxs = np.arange(len(labels))
    num_labels = max(labels) + 1
    bootstrap_idxs = np.zeros_like(idxs)
    ptr = 0
    for i in range(num_labels):
        strat = idxs[labels == i]
        bootstrap_idxs[ptr:ptr + len(strat)] = np.random.choice(strat, len(strat), replace=True)
        ptr += len(strat)
    return bootstrap_idxs


def split_hold_out_set(labels, val_size=1000, seed=None):
    """
    Stratify a held-out set for a dataset.

    Parameters
    ----------
    labels: (m,)-length array of integers in {0, ..., num_labels - 1}, where m is number of data pts
    val_size: integer, number of data points to withhold
    seed: integer, if specified will seed before randomization for reproducible splits

    Returns
    -------
    tr_idxs: (m - val_size,)-length array of indices for training set
    val_idxs: (val_size,)-length array of indices for validation set
    """
    if seed is not None:
        np.random.seed(seed)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    idxs = np.arange(len(labels))
    np.random.shuffle(idxs)
    tr_idxs = np.zeros(len(labels) - val_size, dtype=np.int32)
    val_idxs = np.zeros(val_size, dtype=np.int32)
    num_labels = max(labels) + 1
    tr_ptr, val_ptr = 0, 0

    for i in range(num_labels):
        tr_strat = idxs[labels == i][val_size // num_labels:]
        val_strat = idxs[labels == i][:val_size // num_labels]
        tr_idxs[tr_ptr:tr_ptr + len(tr_strat)] = tr_strat
        val_idxs[val_ptr:val_ptr + len(val_strat)] = val_strat
        tr_ptr += len(tr_strat)
        val_ptr += len(val_strat)
    return tr_idxs, val_idxs
