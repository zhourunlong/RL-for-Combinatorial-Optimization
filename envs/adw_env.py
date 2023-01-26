from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from .base_env import BaseEnv
import torch
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')


class ADWEnv(BaseEnv):
    def __init__(self, device, distr_type, distr_granularity, batch_size, **kwargs):
        super().__init__(device, distr_type, batch_size)
        self.gran = int(distr_granularity)

    def move_device(self, device):
        self.device = device
        self.Fv.to(self.device)

    def set_curriculum_params(self, param):
        self.plot_states = None
        [self.n, self.m, self.V] = param
        self.horizon = self.n
        self.bs = self.bs_per_horizon * (self.horizon + 1)
        if self.type == "uniform":
            self.Fv = torch.ones(
                (self.gran,), dtype=torch.double, device=self.device)
        elif self.type == "random":
            self.Fv = torch.rand(
                (self.gran,), dtype=torch.double, device=self.device)
        else:
            self.Fv = torch.rand(
                (self.gran,), dtype=torch.double, device=self.device)
        self.new_instance()

        self.clean()

    @property
    def curriculum_params(self):
        return [self.n, self.m, self.V]

    @property
    def action_size(self):
        return self.m + 1

    def sample_distr(self, F):
        if (self.type != "special"):
            interval = torch.multinomial(
                F, self.bs * self.n * (self.m), replacement=True)
            sample = interval.double() + torch.rand(self.bs * self.n * (self.m),
                                                    dtype=torch.double, device=self.device)
            return sample.view(self.bs, self.n, self.m) / self.gran
        else:
            sample = torch.rand(self.bs * self.n * self.m,
                                dtype=torch.double, device=self.device)
            sample = 0.4 * (sample < 0.9) + \
                (4 * (sample - 0.9) + 0.6) * (sample >= 0.9)
            return sample.view(self.bs, self.n, self.m)

    def new_instance(self):
        self.i = 0
        self.total_revenue = torch.zeros(
            (self.bs,), dtype=torch.double, device=self.device)
        self.v = torch.cat((self.sample_distr(self.Fv), torch.zeros(
            self.bs, self.n, 1, device=self.device, dtype=torch.double)), 2)
        self.sum_v = torch.zeros(
            (self.bs, (self.m + 1)), dtype=torch.double, device=self.device)
        self.active = torch.ones(
            (self.bs), dtype=torch.double, device=self.device)
        self.B = torch.ones_like(self.sum_v)

    def get_state(self):
        state = [torch.full((self.bs,), (self.i + 1) / self.n,
                            dtype=torch.double, device=self.device)]
        for j in range(0, self.m):
            state.append(self.v[:, self.i, j])
            state.append(self.B[:, j] - self.sum_v[:, j])
        return torch.stack(state, dim=1)

    def get_reference_action(self):
        ref_action = 2
        if ref_action == 0:
            return torch.argmax((self.v[:, self.i, :] * (math.e ** (self.sum_v / self.B))), 1).double()
        elif ref_action == 1:
            return torch.argmax(1 / (0.5 - self.v[:, self.i, :]) * ((self.B - self.sum_v) > self.v[:, self.i, :]), 1).double()
        else:
            return torch.argmax((self.v[:, self.i, :] / (self.B - self.sum_v)) * (self.v[:, self.i, :] <= (self.B - self.sum_v)), 1).double()

    def get_reward(self, action):
        bs_range = torch.arange(
            0, self.bs, device=self.device, dtype=torch.long)
        action = action.long()
        valid = torch.zeros(self.bs, self.m + 1,
                            device=self.device, dtype=torch.double)
        valid[bs_range, action] = 1 * self.active[bs_range] * \
            (self.v[bs_range, self.i, action] +
             self.sum_v[bs_range, action] < self.B[bs_range, action])
        addValue = self.v[:, self.i, :] * valid
        self.sum_v += addValue
        addValue = addValue[bs_range, action]
        self.total_revenue += addValue

        win = self.total_revenue >= self.V
        self.total_revenue[win] = self.V
        rwd = self.active * win
        ract = self.active.clone()
        self.active *= (1 - win.double())
        self.i += 1
        return rwd, ract

    def clean(self):
        del self.v, self.sum_v, self.plot_states
        self.plot_states = None

    def get_plot_states(self):
        print("skip")

    def plot_prob_figure(self, agent, pic_dir):
        print("skip")