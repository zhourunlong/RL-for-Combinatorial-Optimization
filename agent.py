from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

class LogLinearAgent(ABC):
    @abstractmethod
    def __init__(self, device, lr, L, G, U):
        self.lr = float(lr)
        self.L = float(L)
        self.G = float(G)
        self.U = float(U)
        self.device = device
        self.theta = torch.zeros((self.d, 1), dtype=torch.double, device=self.device)
    
    @abstractmethod
    def move_device(self, device):
        pass

    @abstractmethod
    def set_curriculum_params(self, param):
        pass

    @abstractmethod
    def clear_params(self):
        pass

    @abstractmethod
    def get_phi_batch(self, states):
        pass

    def get_logits(self, states):
        phi = self.get_phi_batch(states)
        return phi @ self.theta

    def get_accept_prob(self, states):
        params = self.get_logits(states)
        return torch.sigmoid(params).view(-1,)
    
    def get_action(self, states):
        params = self.get_logits(states)
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)

        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)
        entropy = -(probs * log_probs).sum(-1)

        action = probs.multinomial(1)

        return action.double().view(-1,), entropy
    
    def query_sa(self, states, actions):
        phi = self.get_phi_batch(states)

        params = phi @ self.theta
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)

        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)

        actions = actions.long().view(states.shape[0], 1)

        prob = probs.gather(1, actions).view(-1,)
        log_prob = log_probs.gather(1, actions).view(-1,)
        grad_logp = (1 - prob).view(-1, 1) * (1 - 2 * actions) * phi

        return log_prob, grad_logp

    def zero_grad(self):
        self.grads = torch.zeros_like(self.theta)
        self.F = torch.zeros((self.d, self.d), dtype=torch.double, device=self.device)
        self.cnt = 0
    
    def store_grad(self, As, grads_logp):
        self.grads += (As.unsqueeze(-1) * grads_logp).mean(0).unsqueeze(-1)
        self.F += (grads_logp.T @ grads_logp) / grads_logp.shape[0]
        self.cnt += 1
    
    def _project(self, x, r):
        norm = torch.norm(x)
        if norm > r:
            x *= r / norm
        return x
    
    def _quad_func(self, A, b, x):
        return x.T @ A @ x - 2 * b.T @ x

    def update_param(self):
        # Simple Project Solver
        ngrads = torch.lstsq(self.grads, self.F + 1e-6 * self.cnt * torch.eye(self.d, dtype=torch.double, device=self.device)).solution[:self.d]
        ngrads = self._project(ngrads, self.G)

        # PGD
        '''
        ngrads = torch.lstsq(self.grads, self.F).solution[:self.d]
        norm = torch.norm(ngrads)
        if norm > self.G:
            ngrads = self._project(ngrads, self.G)
            for _ in range(10000):
                ngrads -= 0.01 * (self.F @ ngrads - self.grads) / self.cnt
                ngrads = self._project(ngrads, self.G)
        '''

        self.theta += self.lr * ngrads



class CSPAgent(LogLinearAgent):
    def __init__(self, device, d0, lr, L, G, U, **kwargs):
        self.d0 = int(d0)
        self.d = self.d0 * 2
        super().__init__(device, lr, L, G, U)
    
    def move_device(self, device):
        self.device = device
        self.theta.to(self.device)

    def set_curriculum_params(self, param):
        self.n = param[0]

    def clear_params(self):
        self.theta = torch.zeros_like(self.theta)

    def get_phi_batch(self, states):
        bs = states.shape[0]

        f_axis = torch.ones((bs, self.d0), dtype=torch.double, device=self.device)
        i_axis = torch.cat((torch.ones((bs, 1), dtype=torch.double, device=self.device), states[:, 1].view(-1, 1)), dim=-1)
        
        fractions = states[:, 0]
        for i in range(1, self.d0):
            f_axis[:, i] = f_axis[:, i - 1] * fractions

        phi = torch.bmm(f_axis.unsqueeze(2), i_axis.unsqueeze(1)).view(bs, -1)
        return phi



class OLKnapsackDecisionAgent(LogLinearAgent):
    def __init__(self, device, d0, lr, L, G, U, **kwargs):
        self.d0 = int(d0)
        self.d = self.d0 ** 5
        super().__init__(device, lr, L, G, U)
    
    def move_device(self, device):
        self.device = device
        self.theta.to(self.device)

    def set_curriculum_params(self, param):
        self.n = param[0]
    
    def clear_params(self):
        self.theta = torch.zeros_like(self.theta)

    def get_phi_batch(self, states):
        bs = states.shape[0]

        ax = torch.zeros((bs, 5, self.d0), dtype=torch.double, device=self.device)
        ax[:, :, 0] = 1
        for i in range(1, self.d0):
            ax[:, :, i] = ax[:, :, i - 1] * states

        t1 = torch.bmm(ax[:, 0, :].unsqueeze(2), ax[:, 1, :].unsqueeze(1)).view(bs, -1, 1)
        t1 = torch.bmm(t1, ax[:, 2, :].unsqueeze(1)).view(bs, -1, 1)
        t2 = torch.bmm(ax[:, 3, :].unsqueeze(2), ax[:, 4, :].unsqueeze(1)).view(bs, 1, -1)
        phi = torch.bmm(t1, t2).view(bs, -1)
        return phi
