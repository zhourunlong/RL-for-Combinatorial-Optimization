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
    
    def move_device(self, device):
        self.device = device
        self.theta.to(self.device)

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
        entropy = -(probs * log_probs).sum(-1)

        action = probs.multinomial(1)

        return action.double().view(-1,), entropy
    
    def query_sa(self, state, action):
        phi = self.get_phi_batch(state)

        params = phi @ self.theta
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)

        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)

        action = action.long().view(state.shape[0], 1)

        prob = probs.gather(1, action).view(-1,)
        log_prob = log_probs.gather(1, action).view(-1,)
        grad_logp = (1 - prob).view(-1, 1) * (1 - 2 * action) * phi

        return log_prob, grad_logp
    
    def get_policy(self):
        phi = self.get_phi_all().view(2 * self.n, -1)
        return torch.sigmoid(phi @ self.theta).view(self.n, 2)

    def zero_grad(self):
        self.grads = torch.zeros_like(self.theta)
        self.F = torch.zeros((self.d, self.d), dtype=torch.double, device=self.device)
        self.cnt = 0
    
    def store_grad(self, As, grads_logp):
        # calc V & Q
        #pi = torch.sigmoid(phis @ self.theta).squeeze(-1)
        #V = torch.zeros((self.n + 1, bs), dtype=torch.double, device=self.device)
        #Q = torch.zeros((self.n, bs), dtype=torch.double, device=self.device)
        #V[-1] = -1
        #for i in range(self.n - 1, -1, -1):
        #    V[i] = pi[i] * rs0s[i] + (1 - pi[i]) * V[i + 1]
        #V[:-1, :] += self.L * entropies
        #for i in range(self.n - 1, -1, -1):
        #    Q[i] = torch.where(actions[i] == 0.0, rs0s[i], V[i + 1]) - self.L * pi[i].log()
        
        # NPG
        self.grads += (As.unsqueeze(-1) * grads_logp).mean(0).unsqueeze(-1)
        self.F += (grads_logp.view(-1, self.d, 1) @ grads_logp.view(-1, 1, self.d)).mean(0)
        self.cnt += 1

        # NPG unif(A)
        #phis *= acts.unsqueeze(-1)
        #grads = (0.5 * (((1 - pi) * rs0s - pi * V[1:,:] + (2 * pi - 1) * V[:-1,:]) * acts).unsqueeze(-1) * phis).mean((0, 1)).unsqueeze(-1)
        #F = ((0.5 * ((1 - pi) ** 2 + pi ** 2).unsqueeze(-1) * phis).view(-1, self.d, 1) @ phis.view(-1, 1, self.d)).mean(0)
        
        # Q-NPG unif(A)
        #grads_logp = acts.unsqueeze(-1) * phis
        #grads = ((rs0s * acts).unsqueeze(-1) * grads_logp).mean((0, 1)).unsqueeze(-1)
        #F = (grads_logp.view(-1, self.d, 1) @ grads_logp.view(-1, 1, self.d)).mean(0)
        
    def update_param(self):
        ngrads = torch.lstsq(self.grads, self.F + 1e-6 * self.cnt * torch.eye(self.d, dtype=torch.double, device=self.device)).solution[:self.d]
        #ngrads = F.pinverse() @ grads

        norm = torch.norm(ngrads)
        if norm > self.W:
            ngrads *= self.W / norm

        self.theta += self.lr * self.n * ngrads

        #project to W ball
        #norm = torch.norm(self.theta)
        #if norm > self.W:
        #    self.theta *= self.W / norm
     
    def get_logits(self, states):
        phi = self.get_phi_batch(states)

        params = torch.matmul(phi, self.theta)

        return params

    def get_accept_prob(self, states):
        params = self.get_logits(states)
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)

        probs = F.softmax(params, dim=-1)

        return probs[:, 0].view(-1,)
