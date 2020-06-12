

class Score(object):
    
    @staticmethod
    def score(q, y):
        """
        Score for forecast q and observation y.

        Parameters
        ----------
        q: (n, k) shaped array for n forecasts over k categories
        y: (n, k) shaped array for n one-hot observations over k categories

        Returns
        -------
        S: (n,) length array of scores
        """
        raise NotImplementedError

    @staticmethod
    def entropy(q):
        """
        Entropy.

        Parameters
        ----------
        q: (n, k) shaped array for n forecasts over k categories

        Returns
        -------
        H: (n,) length array of entropies
        """
        raise NotImplementedError

    @staticmethod
    def divergence(q, p):
        """
        Divergence from p to q (not necessarily symmetric).

        Parameters
        ----------
        q: (n, k) shaped array for n forecasts over k categories
        p: (n, k) shaped array for n forecasts over k categories

        Returns
        -------
        D: (n,) length array of pairwise divergences
        """
        raise NotImplementedError

    @staticmethod
    def resolution(q, p, bin_cnts):
        """
        Resolution of discretized forecasts, where q represents observations and p the marginal.

        Parameters
        ----------
        q: (d, k) shaped array for d discretized ground-truth forecasts over k categories
        p: (d, k) shaped array for d discretized marginal forecasts over k categories
        bin_cnts: (d,) length array 

        Returns
        -------
        R: scalar representing resolution
        """
        raise NotImplementedError

    @staticmethod
    def reliability(q, p, bin_cnts):
        """
        Reliability of discretized forecasts, where q represents observations and p predictions.

        Parameters
        ----------
        q: (d, k) shaped array for d discretized ground-truth forecasts over k categories
        p: (d, k) shaped array for d discretized predicted forecasts over k categories
        bin_cnts: (d,) length array 

        Returns
        -------
        R: scalar representing reliability
        """
        raise NotImplementedError

    @staticmethod
    def uncertainty(q, num_observations):
        """
        Uncertainty of the marginal forecast q. 

        Parameters
        ----------
        q: (1, k) shaped array for a marginal forecast over k categories
        num_observations: scalar
        """
        raise NotImplementedError


