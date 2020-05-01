import torch
from torch.distributions import Beta


def mixup_batch(x, y, alpha=0.3):
    """
    Mixup for a batch of data [Zhang et al. ICLR 2018].
    
    Sample lambda ~ Beta(alpha, alpha), return a convex combination of inputs, and permuted inputs.
    This is implemented by doubling the size of the batch and adding sample weights.
    Notice that if lambda = 0 or lambda = 1 that's identical to the vanilla (no mixup) training.

    Notes
    -----
    The authors find that alpha in [0.1, 0.4] generally yielded best performance on ImageNet.

    Returns
    -------
    x: (batch_size, ?)
    y: (batch_size,)
    w: (batch_size,)
    """
    batch_size = len(x)
    lambd = Beta(alpha, alpha).sample()
    idxs = torch.randperm(batch_size, device=x.device, requires_grad=False)

    x = lambd * x + (1 - lambd) * x[idxs]
    x = torch.cat((x, x))
    y = torch.cat((y, y[idxs]))
    w = torch.cat((2. * lambd * torch.ones(batch_size, device=x.device, dtype=x.dtype), 
                   2. * (1 - lambd) * torch.ones(batch_size, device=x.device, dtype=x.dtype)))
    return x, y, w

