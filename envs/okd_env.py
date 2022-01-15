from .base_env import BaseEnv
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import Axes3D

class OKDEnv(BaseEnv):
    def __init__(self, device, distr_type, distr_granularity, batch_size, **kwargs):
        super().__init__(device, distr_type, batch_size)
        self.gran = int(distr_granularity)
    
    def move_device(self, device):
        self.device = device
        self.Fv.to(self.device)
        self.Fs.to(self.device)
    
    def set_curriculum_params(self, param):
        self.plot_states = None
        [self.n, self.B, self.T] = param
        self.horizon = self.n
        self.bs = self.bs_per_horizon * (self.horizon + 1)
        if self.type == "uniform":
            self.Fv = torch.ones((self.gran,), dtype=torch.double, device=self.device)
            self.Fs = torch.ones_like(self.Fv)
        else:
            self.Fv = torch.rand((self.gran,), dtype=torch.double, device=self.device)
            self.Fs = torch.rand((self.gran,), dtype=torch.double, device=self.device)
        self.new_instance()
        self.calc_ref_r()
        self.clean()
    
    @property
    def curriculum_params(self):
        return [self.n, self.B, self.T]
    
    @property
    def action_size(self):
        return 2
    
    def sample_distr(self, F):
        interval = torch.multinomial(F, self.bs * self.n, replacement=True)
        sample = interval.double() + torch.rand(self.bs * self.n, dtype=torch.double, device=self.device)
        return sample.view(self.bs, self.n) / self.gran

    def new_instance(self):
        self.i = 0
        self.v = self.sample_distr(self.Fv)
        self.s = self.sample_distr(self.Fs)
        self.sum_v = torch.zeros((self.bs,), dtype=torch.double, device=self.device)
        self.sum_s = torch.zeros_like(self.sum_v)
        self.active = torch.ones_like(self.sum_v)

    def get_state(self):
        return torch.stack((self.v[:, self.i], self.s[:, self.i], torch.full((self.bs,), (self.i + 1) / self.n, dtype=torch.double, device=self.device), self.sum_s / self.B, self.sum_v / self.T), dim=1)
    
    def get_reference_action(self):
        return (self.v[:, self.i] < self.r * self.s[:, self.i]).double()
    
    def get_reward(self, action):
        pickable = (self.sum_s + self.s[:, self.i]) <= self.B
        valid = self.active * (1 - action) * pickable
        #die = self.active * (1 - action) * (1 - pickable.double())
        self.sum_s += valid * self.s[:, self.i]
        self.sum_v += valid * self.v[:, self.i]
        win = self.sum_v >= self.T
        self.sum_v[win] = self.T
        rwd = self.active * win
        ract = self.active.clone()
        self.active *= (1 - win.double()) #* (1 - die.double())
        self.i += 1
        return rwd, ract
    
    def clean(self):
        del self.v, self.s, self.sum_s, self.sum_v, self.plot_states
        self.plot_states = None
    
    def calc_ref_r(self):
        def calc(r):
            sum = torch.zeros((self.bs,), dtype=torch.double, device=self.device)
            rwd = torch.zeros_like(sum)
            for i in range(self.horizon):
                action = (self.v[:, i] < r * self.s[:, i]).double()
                valid = (1 - action) * ((sum + self.s[:, i]) <= self.B)
                sum += valid * self.s[:, i]
                rwd += valid * self.v[:, i]
            
            return rwd.mean().item()

        l, r = 0, 10
        for _ in range(20):
            m1, m2 = (2 * l + r) / 3, (l + 2 * r) / 3
            c1, c2 = calc(m1), calc(m2)
            if c1 > c2:
                r = m2
            else:
                l = m1
        self.r = l
    
    def get_plot_states(self):
        if self.plot_states is not None:
            return self.plot_states
        
        x = torch.linspace(0.02, 1, 50, device=self.device)
        f = torch.linspace(0.1, 0.9, 9, device=self.device)
        r = torch.linspace(0.1, 0.9, 6, device=self.device)
        v, s, f, r, q = torch.meshgrid(x, x, f, r, torch.zeros((1,), device=self.device))
        self.plot_states = torch.stack((v.reshape(-1,), s.reshape(-1,), f.reshape(-1,), r.reshape(-1,), q.reshape(-1,)), dim=1)

        return self.plot_states
    
    def plot_prob_figure(self, agent, pic_dir):
        fig = plt.figure(figsize=(22, 25))
        color_map = "viridis"
        
        acc = agent.get_accept_prob(self.get_plot_states()).view(50, 50, 9, 6, 1).cpu().numpy()
        x = np.linspace(0.02, 1, 50)
        X, Y = np.meshgrid(x, x, indexing="ij")

        for t in range(9):
            ax = fig.add_subplot(3, 3, t + 1, projection='3d')
            ax.set_title("i/n = 0.%d" % (t + 1))

            for i in range(6):
                z = i / 6
                ax.contourf(X, Y, z + 0.02 / 6 * acc[:, :, t, i, 0], zdir='z', levels=50, cmap=color_map, norm=matplotlib.colors.Normalize(vmin=z, vmax=z + 0.02 / 6))

            ax.set_xlim3d(0, 1)
            ax.set_xlabel("v")
            ax.set_ylim3d(0, 1)
            ax.set_ylabel("s")
            ax.set_zlim3d(-0.01, 1.01 - 1 / 6)
            ax.set_zlabel("sum/B")
            ax.invert_zaxis()
            ax.view_init(-170, 60)

        fig.subplots_adjust(wspace=0, hspace=0, right=0.9)
        position = fig.add_axes([0.92, 0.4, 0.015, 0.2])
        cb = fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=color_map), cax=position)

        plt.savefig(pic_dir, bbox_inches="tight")
        plt.close()