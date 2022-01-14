from .base_agent import LogLinearAgent
import torch

class SPAgent(LogLinearAgent):
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
