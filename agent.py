import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable

class CSPLogLinearAgent():
    def __init__(self, lr, d0, L, W, device):
        self.lr = lr
        self.d0 = d0
        self.d = d0 * 2
        self.L = L
        self.W = W
        self.device = device

        self.theta = torch.zeros((self.d, 1), dtype=torch.double, device=self.device)
    
    def update_n(self, n):
        self.n = n

    def get_phi_batch(self, states):
        bs = states.shape[0]

        f_axis = torch.ones((bs, self.d0), dtype=torch.double, device=self.device)
        i_axis = torch.cat((torch.ones((bs, 1), dtype=torch.double, device=self.device), states[:, 1].view(-1, 1)), dim=-1)
        
        fractions = states[:, 0]
        for i in range(1, self.d0):
            f_axis[:, i] = f_axis[:, i - 1] * fractions

        phi = torch.bmm(f_axis.unsqueeze(2), i_axis.unsqueeze(1)).view(bs, -1)
        return phi
    
    def get_phi_all(self):
        f = torch.arange(1, self.n + 1, dtype=torch.double, device=self.device) / self.n
        f = torch.stack((f, f), dim=1).view(-1,)

        x = torch.tensor([0, 1], dtype=torch.double, device=self.device)
        x = x.repeat(self.n, 1).view(-1,)

        phi = self.get_phi_batch(torch.stack((f, x), dim=1))
        return phi.view(self.n, 2, -1)
    
    def get_action(self, state):
        phi = self.get_phi_batch(state)

        params = phi @ self.theta
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)

        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)

        action = probs.multinomial(1)
        prob = probs.gather(1, action).view(-1,)
        log_prob = log_probs.gather(1, action).view(-1,)
        entropy = -(probs * log_probs).sum(-1)

        action = action.double()

        #grad_logp = (1 - prob).view(-1, 1) * (1 - 2 * action) * phi

        return action.view(-1,), prob, entropy
    
    def get_policy(self):
        phi = self.get_phi_all().view(2 * self.n, -1)
        return torch.sigmoid(phi @ self.theta).view(self.n, 2)
        
    def update_param(self, states, actions, probs, entropies, rs0s, acts):
        bs = states[0].shape[0]
        
        states = torch.cat(states)

        phis = self.get_phi_batch(states).view(self.n, bs, -1)
        pi = torch.sigmoid(phis @ self.theta).squeeze(-1)

        V = torch.zeros((self.n + 1, bs), dtype=torch.double, device=self.device)
        Q = torch.zeros((self.n, bs), dtype=torch.double, device=self.device)
        V[-1] = -1
        for i in range(self.n - 1, -1, -1):
            V[i, :] = pi[i, :] * rs0s[i] + (1 - pi[i, :]) * V[i + 1, :] + self.L * entropies[i]
            Q[i] = torch.where(actions[i] == 0.0, rs0s[i], V[i + 1]) - self.L * pi[i, :].log()

        rs0s = torch.stack(rs0s)
        acts = torch.stack(acts)
        probs = torch.stack(probs)
        actions = torch.stack(actions)
        
        # NPG
        grads_logp = (acts * (1 - probs) * (1 - 2 * actions)).unsqueeze(-1) * phis
        grads = ((Q * acts).unsqueeze(-1) * grads_logp).mean((0, 1)).unsqueeze(-1)
        F = (grads_logp.view(-1, self.d, 1) @ grads_logp.view(-1, 1, self.d)).mean(0)

        # NPG unif(A)
        #phis *= acts.unsqueeze(-1)
        #grads = (0.5 * (((1 - pi) * rs0s - pi * V[1:,:] + (2 * pi - 1) * V[:-1,:]) * acts).unsqueeze(-1) * phis).mean((0, 1)).unsqueeze(-1)
        #F = ((0.5 * ((1 - pi) ** 2 + pi ** 2).unsqueeze(-1) * phis).view(-1, self.d, 1) @ phis.view(-1, 1, self.d)).mean(0)
        
        # NPG unif(A) importance sampling (NOT DONE!)
        #grads_logp = (acts * (1 - probs) * (1 - 2 * actions)).unsqueeze(-1) * phis
        #phis *= acts.unsqueeze(-1)
        #grads = ((acts / (2 * probs)).unsqueeze(-1) * grads_logp * Q).mean((0, 1)).unsqueeze(-1)
        #F = ((0.5 * ((1 - pi) ** 2 + pi ** 2).unsqueeze(-1) * phis).view(-1, self.d, 1) @ phis.view(-1, 1, self.d)).mean(0)

        # Q-NPG unif(A)
        #grads_logp = acts.unsqueeze(-1) * phis
        #grads = ((rs0s * acts).unsqueeze(-1) * grads_logp).mean((0, 1)).unsqueeze(-1)
        #F = (grads_logp.view(-1, self.d, 1) @ grads_logp.view(-1, 1, self.d)).mean(0)

        C = 1e-6
        ngrads = torch.lstsq(grads, F + C * torch.eye(self.d, dtype=torch.double, device=self.device)).solution[:self.d]
        #ngrads = F.pinverse() @ grads

        norm = torch.norm(ngrads)
        if norm > self.W:
            ngrads *= self.W / norm

        self.theta += self.lr * self.n * ngrads

        #project to W ball
        #norm = torch.norm(self.theta)
        #if norm > self.W:
        #    self.theta *= self.W / norm
        
        #print(torch.cat((self.theta, grads), dim=1))
        
        #print("grad norm", grad_norm.item(), "norm", norm.item())
     
    def get_logits(self, states):
        phi = self.get_phi_batch(states)

        params = torch.matmul(phi, self.theta)

        return params

    def get_accept_prob(self, states):
        params = self.get_logits(states)
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)

        probs = F.softmax(params, dim=-1)

        return probs[:, 0].view(-1,)
