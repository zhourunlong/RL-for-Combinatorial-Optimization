from abc import ABC, abstractmethod
import torch
import numpy as np
from math import *
from fractions import Fraction
from decimal import Decimal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import Axes3D

class BaseEnv(ABC):
    @abstractmethod
    def __init__(self, device, distr_type, batch_size):
        self.type = distr_type
        self.device = device
        self.bs_per_horizon = int(batch_size)
    
    @abstractmethod
    def move_device(self, device):
        pass
    
    @abstractmethod
    def set_curriculum_params(self, param):
        pass

    @property
    @abstractmethod
    def curriculum_params(self):
        pass

    @property
    @abstractmethod
    def action_size(self):
        pass

    @abstractmethod
    def new_instance(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_reference_action(self):
        pass
    
    @abstractmethod
    def get_reward(self, action):
        pass

    @abstractmethod
    def clean(self):
        pass

    @abstractmethod
    def plot_prob_figure(self, agent, pic_dir):
        pass

    @property
    def cnt_samples(self):
        return self.horizon * self.bs_per_horizon

class CSPEnv(BaseEnv):
    def __init__(self, device, distr_type, batch_size, **kwargs):
        super().__init__(device, distr_type, batch_size)
    
    def move_device(self, device):
        self.device = device
        self.probs.to(self.device)
    
    def set_curriculum_params(self, param):
        [self.n] = param
        self.horizon = self.n
        self.bs = self.bs_per_horizon * (self.horizon + 1)
        if self.type == "uniform":
            self.probs = 1 / torch.arange(1, self.n + 1, dtype=torch.double, device=self.device)
        else:
            tmp = 1 / torch.arange(2, self.n + 1, dtype=torch.double, device=self.device)
            self.probs = torch.cat((torch.ones((1,), dtype=torch.double, device=self.device), tmp.pow(0.25 + 2 * torch.rand(self.n - 1, dtype=torch.double, device=self.device))))
        self.calc_opt_policy()
    
    @property
    def curriculum_params(self):
        return [self.n]
    
    @property
    def action_size(self):
        return 2

    def new_instance(self):
        self.i = 0
        self.v = self.probs.repeat(self.bs, 1).bernoulli()
        self.argmax = torch.argmax(self.v + torch.arange(self.n, dtype=torch.double, device=self.device) * 1e-5, 1)
        self.active = torch.ones((self.bs,), dtype=torch.double, device=self.device)

    def get_state(self):
        return torch.stack((torch.full((self.bs,), (self.i + 1) / self.n, dtype=torch.double, device=self.device), self.v[:, self.i].double()), dim=1)
    
    def get_reference_action(self):
        return 1 - self.opt_policy[self.i, self.v[:, self.i].long()]
        
    def get_reward(self, action):
        eq = (self.argmax == self.i).double()
        self.i += 1
        if self.i == self.n:
            return self.active * (1 - action) * eq, self.active
        ret = self.active * (1 - action) * eq
        ract = self.active.clone()
        self.active *= action
        return ret, ract

    def calc_opt_policy(self):
        n = self.n
        probs = self.probs.cpu()

        Q = [[[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]] for _ in range(n + 1)]
        V = [[Fraction(0), Fraction(0)] for _ in range(n + 1)]

        Q[n][0][0] = Fraction(-1)
        Q[n][0][1] = Fraction(-1)
        V[n][0] = Fraction(-1)

        Q[n][1][0] = Fraction(1)
        Q[n][1][1] = Fraction(-1)
        V[n][1] = Fraction(1)

        pi_star = torch.zeros((n, 2))
        pi_star[n - 1, 1] = 1

        prob_max = Fraction(1)
        for i in range(n - 1, 0, -1):
            p_i = Fraction(Decimal.from_float(np.float(probs[i])))
            prob_max *= 1 - p_i

            Q[i][0][0] = Fraction(-1)
            Q[i][0][1] = (1 - p_i) * V[i + 1][0] + p_i * V[i + 1][1]
            
            Q[i][1][0] = 2 * prob_max - 1
            Q[i][1][1] = Q[i][0][1]

            V[i][0] = max(Q[i][0][0], Q[i][0][1])
            V[i][1] = max(Q[i][1][0], Q[i][1][1])

            if V[i][1] == Q[i][1][0]:
                pi_star[i - 1, 1] = 1

        self.opt_policy = pi_star.to(self.device)

    def clean(self):
        del self.v, self.argmax, self.active

    def plot_prob_figure(self, agent, pic_dir):
        n = 1000
        f = torch.arange(0, n + 1, dtype=torch.double, device=self.device) / n
        states = torch.stack((f, torch.ones_like(f)), dim=1)
        max_acc = agent.get_accept_prob(states).cpu().numpy()
        states = torch.stack((f, torch.zeros_like(f)), dim=1)
        non_max_acc = agent.get_accept_prob(states).cpu().numpy()

        fig, ax = plt.subplots(figsize=(20, 20))

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        x = np.arange(0, n + 1) / n
        ax.plot(x, max_acc, label="P[Accept|PrefMax]")
        ax.plot(x, non_max_acc, label="P[Accept|NotPrefMax]")

        pi_star = self.opt_policy
        ax.plot(np.arange(0, self.n + 1) / self.n, np.concatenate(([0], pi_star[:, 1].cpu().numpy())), label="Optimal")

        ax.set_title("Plot of Policy", fontsize=40)
        ax.set_xlabel("Time", fontsize=30)
        ax.set_ylabel("Prob", fontsize=30)
        ax.legend(loc="best", fontsize=30)

        plt.savefig(pic_dir)
        plt.close()



class OLKnapsackDecisionEnv(BaseEnv):
    def __init__(self, device, distr_type, distr_granularity, batch_size, **kwargs):
        super().__init__(device, distr_type, batch_size)
        self.gran = int(distr_granularity)
    
    def move_device(self, device):
        self.device = device
        self.Fv.to(self.device)
        self.Fs.to(self.device)
    
    def set_curriculum_params(self, param):
        self.plot_states = None
        [self.n, self.B, self.V] = param
        self.horizon = self.n
        self.bs = self.bs_per_horizon * (self.horizon + 1)
        if self.type == "uniform":
            self.Fv = torch.ones((self.gran,), dtype=torch.double, device=self.device) / self.gran
            self.Fs = torch.ones((self.gran,), dtype=torch.double, device=self.device) / self.gran
        else:
            self.Fv = torch.rand((self.gran,), dtype=torch.double, device=self.device) / self.gran
            self.Fs = torch.rand((self.gran,), dtype=torch.double, device=self.device) / self.gran
        self.new_instance()
        self.calc_ref_r()
        self.clean()
    
    @property
    def curriculum_params(self):
        return [self.n, self.B, self.V]
    
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
        return torch.stack((self.v[:, self.i], self.s[:, self.i], torch.full((self.bs,), (self.i + 1) / self.n, dtype=torch.double, device=self.device), self.sum_s / self.B, self.sum_v / self.V), dim=1)
    
    def get_reference_action(self):
        return (self.v[:, self.i] < self.r * self.s[:, self.i]).double()
    
    def get_reward(self, action):
        pickable = (self.sum_s + self.s[:, self.i]) <= self.B
        valid = self.active * (1 - action) * pickable
        #die = self.active * (1 - action) * (1 - pickable.double())
        self.sum_s += valid * self.s[:, self.i]
        self.sum_v += valid * self.v[:, self.i]
        win = self.sum_v >= self.V
        self.sum_v[win] = self.V
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