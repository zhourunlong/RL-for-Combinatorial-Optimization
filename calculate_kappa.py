import os, sys
import torch

def calc_distr(probs, policy):
    pr_rej = probs * (1 - policy[:, 1]) + (1 - probs) * (1 - policy[:, 0])
    df = pr_rej.cumprod(dim=0)
    df = torch.cat((torch.ones((1,), dtype=torch.double, device=df.device), df[:-1]))
    df1 = df * probs
    dfx = torch.stack((df - df1, df1), dim=1)
    return dfx

def calc_sigma(probs, policy_d, policy_t, phi):
    d = calc_distr(probs, policy_d)
    w = (1 - policy_t) ** 2 + policy_t ** 2
    raw = phi.unsqueeze(-1) @ phi.unsqueeze(-2)
    sigma = ((d * w).view(-1, 2, 1, 1) * raw).sum((0, 1))
    return sigma
    
def calc_kappa(probs, policy_star, policy_t, phi):
    sigma_star = calc_sigma(probs, policy_star, policy_t, phi)
    sigma_t = calc_sigma(probs, policy_t, policy_t, phi)
    
    S, U = torch.symeig(sigma_t, eigenvectors=True)

    sqinv = 1 / S.sqrt()
    above_cutoff = S > 1e-12
    sqinv = sqinv[above_cutoff]
    U = U[:, above_cutoff]
    
    st = U @ torch.diag(sqinv) @ U.T

    e = torch.symeig(st @ sigma_star @ st)[0]
    return e[-1]