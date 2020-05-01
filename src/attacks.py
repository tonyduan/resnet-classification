import torch
from torch.autograd import grad


def project_onto_ball(x, eps, p=float("inf")):
    """
    Note that projection onto inf-norm and 2-norm take O(d) time, and projection onto 1-norm
    takes O(dlogd) using the sorting-based algorithm given in [Duchi et al. 2008].

    Parameters
    ----------
    x: (batch_size, ?) variable dimensional tensor input
    eps: float for size of ball
    p: float for p-norm
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    assert not torch.isnan(x).any()
    if p == float("inf"):
        x = x.clamp(-eps, eps)
    elif p == 2:
        x = x.renorm(p=2, dim=0, maxnorm=eps)
    elif p == 1:
        mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
        mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
        cumsum = torch.cumsum(mu, dim=1)
        arange = torch.arange(1, x.shape[1] + 1, device=x.device)
        rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
        theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
        proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
        x = mask * x + (1 - mask) * proj * torch.sign(x)
    else:
        raise ValueError("Can only project onto 1,2,inf norm balls.")
    return x.view(original_shape)


def pgd_attack(model, x, y, eps, steps=20, adv=float("inf"), clamp=(0., 1.)):
    """
    Attack a model with PGD [Madry et al. ICLR 2017].
    """
    step_size = 2 * eps / steps
    x_orig = x.clone()
    x.requires_grad = True

    for _ in range(steps):
        loss = model.loss(x, y).mean()
        grads = grad(loss, x, only_inputs=True)[0].reshape(x.shape[0], -1)
        if adv == 1:
            keep_vals = torch.kthvalue(grads.abs(), k=grads.shape[1] * 15 // 16, dim=1).values
            grads[torch.abs(grads) < keep_vals.unsqueeze(1)] = 0
            grads = torch.sign(grads)
            grads_norm = torch.norm(grads, dim=1, p=1)
            grads = grads / (grads_norm.unsqueeze(1) + 1e-8)
        elif adv == 2:
            grads_norm = torch.norm(grads, dim=1, p=2)
            grads = grads / (grads_norm.unsqueeze(1) + 1e-8)
        elif adv == float("inf"):
            grads = torch.sign(grads)
        else:
            raise ValueError
        diff = x + step_size * grads.reshape(x.shape) - x_orig
        diff = project_onto_ball(diff, eps, adv)
        x = (x_orig + diff).clamp(*clamp)

    return x.detach()


def fgsm_attack(model, x, y, eps, clamp=(0., 1.), alpha=1.):
    """
    FGSM attack for the L-inf adversary [Wong et al. ICLR 2020].

    Notes
    -----
    The authors found alpha in [0.75, 1.25] work well; too high results in catastrophic overfitting.
    """
    step_size = alpha * eps
    diff = (2. * torch.rand_like(x) - 1.) * eps
    diff.requires_grad = True
    
    loss = model.loss(x + diff, y).mean()
    grads = grad(loss, diff, only_inputs=True)[0]
    diff = diff + step_size * torch.sign(grads)
    diff = project_onto_ball(diff, eps, p=float("inf"))
    x = (x + diff).clamp(*clamp)

    return x.detach()

