import numpy as np
import scipy as sp
import scipy.special
from src.scores.score import Score


class NLLScore(Score):

    @staticmethod
    def score(q, y, logits=False):
        if logits:
            return -np.sum(q * y, axis=1)
        else:
            return -np.sum(sp.special.xlogy(y, q), axis=1)

    @staticmethod
    def entropy(q):
        return np.sum(sp.special.entr(q), axis=1)

    @staticmethod
    def divergence(q, p):
        return np.sum(sp.special.rel_entr(q, p), axis=1)

    @staticmethod
    def resolution(q, p, bin_cnts):
        """
        Adjustment follows [Brocker, 2012].
        """
        adjustment = 0.5 * (q.shape[0] - 1) * (q.shape[1] - 1) / np.sum(bin_cnts)
        adjustment = 0.
        return np.average(NLLScore.divergence(q, p), weights=bin_cnts) - adjustment

    @staticmethod
    def reliability(q, p, bin_cnts):
        """
        Adjustment follows [Brocker, 2012].
        """
        adjustment = 0.5 * q.shape[0] * (q.shape[1] - 1) / np.sum(bin_cnts)
        adjustment = 0.
        return np.average(NLLScore.divergence(q, p), weights=bin_cnts) - adjustment

    @staticmethod
    def uncertainty(q, num_observations):
        """
        Adjustment follows [Brocker, 2012].
        """
        adjustment = 0.5 * (q.shape[1] - 1) / num_observations
        adjustment = 0.
        return NLLScore.entropy(q).squeeze() + adjustment


