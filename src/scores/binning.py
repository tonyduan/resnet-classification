import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering


def discretize_multivar(q, p, n_bins=20):
    """
    Partition the simplex with K-Means clustering.

    Notice that we can recover the clusters by running something like the following.

    >>> obs, pred, bin_cnts = discretize_multivar(labels_one_hot, preds, n_bins=100)
    >>> unique_bins = np.unique(np.round(pred, 3), axis=0)
    """
    #kmeans = KMeans(n_clusters=n_bins, init="k-means++")
    #kmeans.fit(p)
    clustering = AgglomerativeClustering(n_clusters=n_bins)
    clustering.fit(p)
    bin_ids = clustering.labels_

    pred = np.vstack([p[bin_ids == i].mean(axis=0) for i in range(n_bins)])
    obs = np.vstack([q[bin_ids == i].mean(axis=0) for i in range(n_bins)])
    bin_cnts = np.bincount(bin_ids, minlength=n_bins)
    assert np.all(bin_cnts > 0)

    return obs, pred, bin_cnts

#
#def discretize_multivar(q, p, n_bins=20):
#    """
#    Recursively partition the simplex by splitting along the dimension with highest variance.
#
#    [Vaicenavicius et al. AISTATS 2019].
#    """
#    bins = [(q, p)]
#    while len(bins) < n_bins:
#
#        breakpoint()
#        variances = [p.var(axis=0) for q, p in bins]
#        idx = np.argmax([np.max(v) for v in variances])
#        q, p = bins.pop(idx)
#        dim = np.argmax(p.var(axis=0))
#        median = np.median(p, axis=0)[dim]
#        q_left, p_left = q[p[:, dim] < median], p[p[:, dim] < median]
#        q_right, p_right = q[p[:, dim] >= median], p[p[:, dim] >= median]
#        if len(q_left) == 0 or len(q_right) == 0:
#            bins.append((q, p))
#            break
#        else:
#            bins.append((q_left, p_left))
#            bins.append((q_right, p_right))
#
#    pred = np.vstack([p.mean(axis=0) for q, p in bins])
#    obs = np.vstack([q.mean(axis=0) for q, p in bins])
#    bin_cnts = np.array([len(q) for q, p in bins])
#    assert np.all(bin_cnts > 0)
#    return pred, obs, bin_cnts
#
