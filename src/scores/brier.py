import numpy as np
import scipy as sp
import scipy.stats
import logging
from src.scores.score import Score


class BrierScore(Score):
    
    @staticmethod
    def score(q, y):
        return np.sum((q - y) ** 2, axis=1)

    @staticmethod
    def entropy(q):
        I = np.eye(q.shape[1])[np.newaxis, :, :]
        Q = np.repeat(q[:, :, np.newaxis], q.shape[1], axis=2)
        H = np.sum(Q.transpose(0, 2, 1) * (Q - I) ** 2, axis=(1, 2))
        return H

    @staticmethod
    def divergence(q, p):
        I = np.eye(q.shape[1])[np.newaxis, :, :]
        Q = np.repeat(q[:, :, np.newaxis], q.shape[1], axis=2)
        P = np.repeat(p[:, :, np.newaxis], p.shape[1], axis=2)
        D = np.sum(Q.transpose(0, 2, 1) * ((P - Q) * (P + Q - 2 * I)), axis=(1, 2))
        return D
 
    @staticmethod
    def resolution(q, p, bin_cnts):
        """
        Adjustment follows [Ferro and Fricker, 2012].
        """
        assert np.all(bin_cnts > 0)
        adjustment = (np.sum(bin_cnts / np.clip(bin_cnts - 1, 1, None) * BrierScore.entropy(q)) /
                      np.sum(bin_cnts) - BrierScore.entropy(p).squeeze() / (np.sum(bin_cnts) - 1))
        return np.average(BrierScore.divergence(q, p), weights=bin_cnts) - adjustment

    @staticmethod
    def reliability(q, p, bin_cnts):
        """
        Adjustment follows [Ferro and Fricker, 2012].
        """
        assert np.all(bin_cnts > 0)
        adjustment = (np.sum(bin_cnts / np.clip(bin_cnts - 1, 1, None) * 
                      BrierScore.entropy(q)) / np.sum(bin_cnts))
        return np.average(BrierScore.divergence(q, p), weights=bin_cnts) - adjustment

    def uncertainty(q, num_observations):
        """
        Adjustment follows [Ferro and Fricker, 2012].
        """
        adjustment = BrierScore.entropy(q).squeeze() / (num_observations - 1)
        return BrierScore.entropy(q).squeeze() + adjustment


def _divergence_naive(q, p):
    assert len(q) == len(p)
    I = np.eye(q.shape[1])
    D = np.zeros(q.shape[0])
    for i in range(len(q)):
        for k in range(q.shape[1]):
            D[i] += q[i][k] * (np.sum((p[i] - I[k]) ** 2) - np.sum((q[i] - I[k]) ** 2))
    return D

def _entropy_naive(q):
    I = np.eye(q.shape[1])
    S = np.zeros(q.shape[0])
    for i in range(len(q)):
        for k in range(q.shape[1]):
            S[i] += q[i][k] * np.sum((q[i] - I[k]) ** 2)
    return S


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    np.random.seed(12345)

    logger.info("Test: entropy and divergence functions versus naive implementations")
    m, n = 10000, 5
    q = sp.stats.dirichlet(np.arange(n) / n + 1).rvs(m)
    Score = BrierScore
    
    np.testing.assert_allclose(Score.entropy(q), _entropy_naive(q))
    np.testing.assert_allclose(Score.divergence(q[1:], q[:-1]), _divergence_naive(q[1:], q[:-1]))
    logger.info(f"  Max Difference: < 1e-7")


