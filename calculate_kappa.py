import os, sys
import torch

def calc_distr(probs, policy):
    pr_rej = probs * (1 - policy[:, 1]) + (1 - probs) * (1 - policy[:, 0])
    df = pr_rej.cumprod(dim=0)
    df = torch.cat((torch.ones((1,), dtype=torch.double, device="cuda"), df[:-1]))
    df1 = df * probs
    dfx = torch.stack((df - df1, df1), dim=1)
    return dfx

def calc_sigma(probs, policy_d, policy_t, phi):
    d = calc_distr(probs, policy_d)
    w = (1 - policy_t) ** 2 + policy_t ** 2
    raw = torch.matmul(phi.unsqueeze(-1), phi.unsqueeze(-2))
    sigma = ((d * w).view(-1, 2, 1, 1) * raw).sum((0, 1))
    return sigma
    
def calc_kappa(probs, policy_star, policy_t, phi):
    sigma_star = calc_sigma(probs, policy_star, policy_t, phi)
    sigma_t = calc_sigma(probs, policy_t, policy_t, phi)
    
    U, S, V = torch.svd(sigma_t, compute_uv=True)
    sqinv = 1 / S.sqrt()
    sqinv[S < 1e-15] = 0
    
    st = torch.matmul(torch.matmul(V, torch.diag(sqinv)), U.T)

    e, _ = torch.symeig(torch.matmul(torch.matmul(st, sigma_star), st))
    return e[-1]