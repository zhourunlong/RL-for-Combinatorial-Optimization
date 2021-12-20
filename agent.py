from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

class LogLinearAgent(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
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
        
    def update_param(self):
        ngrads = torch.lstsq(self.grads, self.F + 1e-6 * self.cnt * torch.eye(self.d, dtype=torch.double, device=self.device)).solution[:self.d]

        norm = torch.norm(ngrads)
        if norm > self.W:
            ngrads *= self.W / norm

        self.theta += self.lr * ngrads



class CSPAgent(LogLinearAgent):
    def __init__(self, device, lr, d0, L, W, **kwargs):
        self.lr = float(lr)
        self.d0 = int(d0)
        self.d = self.d0 * 2
        self.L = float(L)
        self.W = int(W)
        self.device = device

        self.theta = torch.zeros((self.d, 1), dtype=torch.double, device=self.device)
    
    def move_device(self, device):
        self.device = device
        self.theta.to(self.device)

    def set_curriculum_params(self, param):
        self.n = param[0]
        self._phi_all = None

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
    
    @property
    def phi_all(self):
        if self._phi_all is not None:
            return self._phi_all
        
        f = torch.arange(1, self.n + 1, dtype=torch.double, device=self.device) / self.n
        f = torch.stack((f, f), dim=1).view(-1,)

        x = torch.tensor([0, 1], dtype=torch.double, device=self.device)
        x = x.repeat(self.n, 1).view(-1,)

        self._phi_all = self.get_phi_batch(torch.stack((f, x), dim=1)).view(self.n, 2, -1)
        return self._phi_all

    @property
    def policy(self):
        phi = self.phi_all.view(2 * self.n, -1)
        return torch.sigmoid(phi @ self.theta).view(self.n, 2)



class OLKnapsackAgent(LogLinearAgent):
    def __init__(self, device, lr, d0, L, W, **kwargs):
        self.lr = float(lr)
        self.d0 = int(d0)
        self.d = self.d0 ** 4
        self.L = float(L)
        self.W = int(W)
        self.device = device

        self.theta = torch.zeros((self.d, 1), dtype=torch.double, device=self.device)
    
    def move_device(self, device):
        self.device = device
        self.theta.to(self.device)

    def set_curriculum_params(self, param):
        self.n = param[0]
    
    def clear_params(self):
        self.theta = torch.zeros_like(self.theta)

    def get_phi_batch(self, states):
        bs = states.shape[0]

        ax = torch.zeros((bs, 4, self.d0), dtype=torch.double, device=self.device)
        ax[:, :, 0] = 1
        for i in range(1, self.d0):
            ax[:, :, i] = ax[:, :, i - 1] * states

        t1 = torch.bmm(ax[:, 0, :].unsqueeze(2), ax[:, 1, :].unsqueeze(1)).view(bs, -1, 1)
        t2 = torch.bmm(ax[:, 2, :].unsqueeze(2), ax[:, 3, :].unsqueeze(1)).view(bs, 1, -1)
        phi = torch.bmm(t1, t2).view(bs, -1)
        return phi
    
    def get_phi_all(self):
        pass
    
    def get_policy(self):
        pass



class OLKnapsackNNAgent():
    def __init__(self, device, lr, d0, L, W, width, depth, **kwargs):
        self.lr = float(lr)
        self.d0 = int(d0)
        self.width = int(width)
        self.L = float(L)
        self.W = int(W)
        self.device = device
        
        self.clear_params()

        self.d = sum(p.numel() for p in self.model.parameters())
    
    def move_device(self, device):
        self.device = device
        self.model.to(self.device)

    def set_curriculum_params(self, param):
        self.n = param[0]
    
    def clear_params(self):
        self.model = nn.Sequential(
            nn.Linear(4, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, 2)
        ).double()
        self.model.to(self.device)

    def get_action(self, states):
        with torch.no_grad():
            params = self.model(states)

            probs = F.softmax(params, dim=-1)
            log_probs = F.log_softmax(params, dim=-1)
        
            entropy = -(probs * log_probs).sum(-1)
            
            action = probs.multinomial(1)

        return action.double().view(-1,), entropy
    
    def query_sa(self, states, actions):
        self.model.zero_grad()

        params = self.model(states)

        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)

        actions = actions.long().view(states.shape[0], 1)

        log_prob = log_probs.gather(1, actions).view(-1,)

        funcs = params.gather(1, actions).view(-1,) - (probs.detach() * params).sum(-1)

        bs = states.shape[0]
        grads_logp = torch.zeros((bs, self.d), dtype=torch.double, device=self.device)
        for i in range(bs):
            for p in self.model.parameters():
                p.grad = None
            funcs[i].backward(retain_graph=True)
            cnt = 0
            for p in self.model.parameters():
                grads_logp[i, cnt:cnt+p.numel()] = p.grad.view(-1,)
                cnt += p.numel()

        return log_prob, grads_logp

    def zero_grad(self):
        self.grads = torch.zeros((self.d, 1), dtype=torch.double, device=self.device)
        self.F = torch.zeros((self.d, self.d), dtype=torch.double, device=self.device)
        self.cnt = 0
    
    def store_grad(self, As, grads_logp):
        self.grads += (As.unsqueeze(-1) * grads_logp).mean(0).unsqueeze(-1)
        self.F += (grads_logp.T @ grads_logp) / grads_logp.shape[0]
        self.cnt += 1

    def update_param(self):
        ngrads = torch.lstsq(self.grads, self.F + 1e-6 * self.cnt * torch.eye(self.d, dtype=torch.double, device=self.device)).solution[:self.d]
        
        norm = torch.norm(ngrads)
        if norm > self.W:
            ngrads *= self.W / norm

        cnt = 0
        for p in self.model.parameters():
            p.data += self.lr * self.n * ngrads[cnt:cnt+p.numel()].view_as(p.data)
            cnt += p.numel()