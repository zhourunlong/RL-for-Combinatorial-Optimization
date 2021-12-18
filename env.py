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
    def __init__(self, **kwargs):
        pass
    
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

    @abstractmethod
    def new_instance(self):
        pass

    @abstractmethod
    def get_state(self):
        pass
    
    @abstractmethod
    def get_reward(self, action):
        pass

    @abstractmethod
    def clean(self):
        pass

    @abstractmethod
    def reference(self):
        pass

    @abstractmethod
    def plot_prob_figure(self, agent, pic_dir):
        pass

    @property
    def cnt_samples(self):
        return self.horizon * self.bs_per_horizon

class CSPEnv(BaseEnv):
    def __init__(self, device, distr_type, rwd_succ, rwd_fail, batch_size, **kwargs):
        self.type = distr_type
        self.device = device
        self.rwd_succ = float(rwd_succ)
        self.rwd_fail = float(rwd_fail)
        self.bs_per_horizon = int(batch_size)
    
    def move_device(self, device):
        self.device = device
        self.probs.to(self.device)
    
    def set_curriculum_params(self, param):
        [self.n] = param
        self.horizon = self.n
        self.bs = self.bs_per_horizon * (self.horizon + 1)
        self._opt_policy = None
        if self.type == "uniform":
            self.probs = 1 / torch.arange(1, self.n + 1, dtype=torch.double, device=self.device)
        else:
            tmp = 1 / torch.arange(2, self.n + 1, dtype=torch.double, device=self.device)
            self.probs = torch.cat((torch.ones((1,), dtype=torch.double, device=self.device), tmp.pow(0.25 + 2 * torch.rand(self.n - 1, dtype=torch.double, device=self.device))))
    
    @property
    def curriculum_params(self):
        return [self.n]

    def new_instance(self):
        self.i = 0
        self.v = self.probs.repeat(self.bs, 1).bernoulli()
        self.argmax = torch.argmax(self.v + torch.arange(self.n, dtype=torch.double, device=self.device) * 1e-5, 1)
        self.active = torch.ones((self.bs,), dtype=torch.double, device=self.device)

    def get_state(self):
        return torch.stack((torch.full((self.bs,), (self.i + 1) / self.n, dtype=torch.double, device=self.device), self.v[:, self.i].double()), dim=1)
    
    def get_reward(self, action):
        eq = (self.argmax == self.i).double()
        raw_reward = eq * self.rwd_succ + (1 - eq) * self.rwd_fail
        self.i += 1
        if self.i == self.n:
            return self.active * ((1 - action) * raw_reward + action * self.rwd_fail), self.active
        ret = self.active * (1 - action) * raw_reward
        ract = self.active.clone()
        self.active *= action
        return ret, ract

    @property
    def opt_policy(self):
        if self._opt_policy is not None:
            return self._opt_policy

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

        pi_star = torch.zeros((n, 2), dtype=torch.double, device=self.device)
        pi_star[n - 1, 1] = 1

        prob_max = Fraction(1)
        for i in range(n - 1, 0, -1):
            p_i = Fraction(Decimal.from_float(np.float(probs[i])))
            prob_max *= 1 - p_i

            Q[i][0][0] = Fraction(-1)
            Q[i][0][1] = (1 - p_i) * V[i + 1][0] + Fraction(p_i) * V[i + 1][1]
            
            Q[i][1][0] = 2 * prob_max - 1
            Q[i][1][1] = Q[i][0][1]

            for j in range(2):
                V[i][j] = max(Q[i][j][0], Q[i][j][1])

            if V[i][1] == Q[i][1][0]:
                pi_star[i - 1, 1] = 1

        self._opt_policy = pi_star
        return self._opt_policy

    def clean(self):
        del self.v, self.argmax, self.active

    def reference(self):
        pi_star = self.opt_policy
        axis = torch.arange(0, self.horizon, dtype=torch.long, device=self.device)
        probs = pi_star[axis, self.v[:, axis].long()].unsqueeze(-1)
        probs = torch.cat([probs, 1 - probs], dim=-1)
        action = probs.view(-1, 2).multinomial(1).view(self.bs, self.horizon)
        active = torch.cat((torch.ones((self.bs, 1), dtype=torch.double, device=self.device), action[:, :-1].cumprod(-1)), -1)
        eq = torch.zeros_like(action)
        baxis = torch.arange(0, self.bs, dtype=torch.long, device=self.device)
        eq[baxis, self.argmax] = 1
        rewards = active * (1 - action) * (eq * self.rwd_succ + (1 - eq) * self.rwd_fail)
        rewards[:, -1] += active[:, -1] * action[:, -1] * self.rwd_fail
        return rewards.sum(1).mean().detach().cpu()

    def calc_distr(self, probs, policy):
        pr_rej = probs * (1 - policy[:, 1]) + (1 - probs) * (1 - policy[:, 0])
        df = pr_rej.cumprod(dim=0)
        df = torch.cat((torch.ones((1,), dtype=torch.double, device=df.device), df[:-1]))
        df1 = df * probs
        dfx = torch.stack((df - df1, df1), dim=1)
        return dfx

    def calc_sigma(self, probs, policy_d, policy_t, phi):
        d = self.calc_distr(probs, policy_d)
        w = (1 - policy_t) * policy_t
        phi = phi.view(-1, phi.shape[-1])
        return phi.T @ torch.diag((d * w).view(-1)) @ phi
        
    def calc_log_kappa(self, policy_t, phi):
        sigma_star = self.calc_sigma(self.probs, self.opt_policy, policy_t, phi)
        sigma_t = self.calc_sigma(self.probs, policy_t, policy_t, phi)

        S, U = torch.symeig(sigma_t, eigenvectors=True)

        pos_eig = S > 0
        sqinv = 1 / S[pos_eig].sqrt()
        U = U[:, pos_eig]
        st = U @ torch.diag(sqinv) @ U.T

        e = torch.symeig(st @ sigma_star @ st.T)[0]
        return log(e[-1])

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

        

class OLKnapsackEnv(BaseEnv):
    def __init__(self, device, distr_type, distr_granularity, batch_size, **kwargs):
        self.type = distr_type
        self.device = device
        self.gran = int(distr_granularity)
        self.bs_per_horizon = int(batch_size)
    
    def move_device(self, device):
        self.device = device
        self.Fv.to(self.device)
        self.Fs.to(self.device)
    
    def set_curriculum_params(self, param):
        self.r = None
        [self.n, self.B] = param
        self.horizon = self.n
        self.bs = self.bs_per_horizon * (self.horizon + 1)
        if self.type == "uniform":
            self.Fv = torch.ones((self.gran,), dtype=torch.double, device=self.device) / self.gran
            self.Fs = torch.ones((self.gran,), dtype=torch.double, device=self.device) / self.gran
        else: # not yet implemented
            self.Fv = torch.ones((self.gran,), dtype=torch.double, device=self.device) / self.gran
            self.Fs = torch.ones((self.gran,), dtype=torch.double, device=self.device) / self.gran
    
    @property
    def curriculum_params(self):
        return [self.n, self.B]
    
    def sample_distr(self, F):
        interval = torch.multinomial(F, self.bs * self.n, replacement=True)
        sample = interval.double() + torch.rand(self.bs * self.n, dtype=torch.double, device=self.device)
        return sample.view(self.bs, self.n) / self.gran

    def new_instance(self):
        self.i = 0
        self.v = self.sample_distr(self.Fv)
        self.s = self.sample_distr(self.Fs)
        self.sum = torch.zeros((self.bs,), dtype=torch.double, device=self.device)
        #self.active = torch.ones_like(self.sum)

    def get_state(self):
        return torch.stack((self.v[:, self.i], self.s[:, self.i], torch.full((self.bs,), (self.i + 1) / self.n, dtype=torch.double, device=self.device), self.sum / self.B), dim=1)
    
    def get_reward(self, action):
        valid = (1 - action) * ((self.sum + self.s[:, self.i]) <= self.B)
        self.sum +=  valid * self.s[:, self.i]
        rwd = valid * self.v[:, self.i]
        act = torch.ones((self.bs,), dtype=torch.double, device=self.device)
        self.i += 1
        return rwd, act
    
    def get_opt_policy(self):
        pass
    
    def clean(self):
        del self.v, self.s, self.sum
    
    def reference(self):
        def calc(r):
            sum = torch.zeros((self.bs,), dtype=torch.double, device=self.device)
            rwd = torch.zeros_like(sum)
            for i in range(self.horizon):
                action = (self.v[:, i] < r * self.s[:, i]).double()
                valid = (1 - action) * ((sum + self.s[:, i]) <= self.B)
                sum += valid * self.s[:, i]
                rwd += valid * self.v[:, i]
            
            return rwd.mean().detach().cpu()

        if self.r is None:
            l, r = 0, 10
            for _ in range(20):
                m1, m2 = (2 * l + r) / 3, (l + 2 * r) / 3
                c1, c2 = calc(m1), calc(m2)
                if c1 > c2:
                    r = m2
                else:
                    l = m1
            self.r = l

        return calc(self.r)
    
    def plot_prob_figure(self, agent, pic_dir):
        fig = plt.figure(figsize=(22, 25))
        lz = 6
        color_map = "viridis"

        for t in range(9):
            ax = fig.add_subplot(3, 3, t + 1, projection='3d')
            ax.set_title("i/n = 0.%d" % (t + 1))

            x = np.linspace(0.01, 1, 100)
            X, Y = np.meshgrid(x, x)

            levels = np.linspace(-1, 1, 40)

            for i in range(lz):
                z = i / lz
                Z = (X ** (i / lz)) * np.cos(Y)
                ax.contourf(X, Y, z + 0.02 / lz * Z, zdir='z', levels=100, cmap=color_map, norm=matplotlib.colors.Normalize(vmin=z, vmax=z + 0.02 / lz))

            ax.set_xlim3d(0, 1)
            ax.set_xlabel("v")
            ax.set_ylim3d(0, 1)
            ax.set_ylabel("s")
            ax.set_zlim3d(1 / lz - 0.01, 1.01)
            ax.set_zlabel("sum/B")
            ax.invert_zaxis()
            ax.view_init(-170, 60)
            

        fig.subplots_adjust(wspace=0, hspace=0, right=0.9)
        position = fig.add_axes([0.92, 0.4, 0.015, 0.2])
        cb = fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=color_map), cax=position)

        plt.savefig("test.jpg", bbox_inches="tight")
        plt.close()