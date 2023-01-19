from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class LogLinearAgent(ABC):
    @abstractmethod
    def __init__(self, device, lr, L, G, U):
        self.lr = float(lr)
        self.L = float(L)
        self.G = float(G)
        self.U = float(U)
        self.device = device
        self.theta = torch.zeros(
            (self.d, 1), dtype=torch.double, device=self.device)

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

    @abstractmethod
    def get_logits(self, states):
        pass
    
    @abstractmethod
    def get_accept_prob(self, states):
        pass

    @abstractmethod
    def get_action(self, states):
        pass

    @abstractmethod
    def query_sa(self, states, actions):
        pass

    def zero_grad(self):
        self.grads = torch.zeros_like(self.theta)
        self.F = torch.zeros(
            (self.d, self.d), dtype=torch.double, device=self.device)
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
        ngrads = torch.lstsq(self.grads, self.F + 1e-6 * self.cnt * torch.eye(
            self.d, dtype=torch.double, device=self.device)).solution[:self.d]
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

        return ngrads
