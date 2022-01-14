from .base_env import BaseEnv
import torch
import numpy as np
from fractions import Fraction
from decimal import Decimal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SPEnv(BaseEnv):
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