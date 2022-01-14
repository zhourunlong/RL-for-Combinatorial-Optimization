from .base_agent import LogLinearAgent
import torch

class OKDAgent(LogLinearAgent):
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
