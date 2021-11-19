from abc import ABC, abstractmethod
import torch
import numpy as np
from fractions import Fraction
from decimal import Decimal

class BaseEnv(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def move_device(self, device):
        pass
    
    @abstractmethod
    def set_n(self, n):
        pass

    @abstractmethod
    def set_bs(self, bs):
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
    def get_opt_policy(self):
        pass

    @abstractmethod
    def clean(self):
        pass

class CSPEnv(BaseEnv):
    def __init__(self, device, distr_type="random", rwd_succ=1, rwd_fail=0, **kwargs):
        self.type = distr_type
        self.device = device
        self.rwd_succ = float(rwd_succ)
        self.rwd_fail = float(rwd_fail)
    
    def move_device(self, device):
        self.device = device
        self.probs.to(self.device)
    
    def set_n(self, n):
        self.n = n
        if self.type == "uniform":
            self.probs = 1 / torch.arange(1, self.n + 1, dtype=torch.double, device=self.device)
        else:
            tmp = 1 / torch.arange(2, self.n + 1, dtype=torch.double, device=self.device)
            self.probs = torch.cat((torch.ones((1,), dtype=torch.double, device=self.device), tmp.pow(0.25 + 2 * torch.rand(self.n - 1, dtype=torch.double, device=self.device))))

    def set_bs(self, bs):
        self.bs = bs

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

    def get_opt_policy(self):
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

        return pi_star

    def clean(self):
        del self.v, self.argmax, self.active

        

class OLKnapsackEnv(BaseEnv):
    def __init__(self, device, distr_type="random", distr_granularity=10, B=5, **kwargs):
        self.type = distr_type
        self.device = device
        self.gran = int(distr_granularity)
        self.B = float(B)
    
    def move_device(self, device):
        self.device = device
        self.Fv.to(self.device)
        self.Fs.to(self.device)
    
    def set_n(self, n):
        self.n = n
        if self.type == "uniform":
            self.Fv = torch.ones((self.gran,), dtype=torch.double, device=self.device) / self.gran
            self.Fs = torch.ones((self.gran,), dtype=torch.double, device=self.device) / self.gran
        else: # not yet implemented
            self.Fv = torch.ones((self.gran,), dtype=torch.double, device=self.device) / self.gran
            self.Fs = torch.ones((self.gran,), dtype=torch.double, device=self.device) / self.gran

    def set_bs(self, bs):
        self.bs = bs
    
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
        return rwd, act
    
    def get_opt_policy(self):
        pass
    
    def clean(self):
        del self.v, self.s, self.sum
    
    def bang_per_buck(self):
        p = 0.15
        k = int(p * self.n)
        sum = torch.zeros((self.bs,), dtype=torch.double, device=self.device)
        rwd = torch.zeros_like(sum)
        bpb = self.v[:, :k] / self.s[:, :k]

        #for i in range(k):
        #    valid = (sum + self.s[:, i]) <= self.B
        #    sum += valid * self.s[:, i]
        #    rwd += valid * self.v[:, i]

        ax = torch.arange(self.bs, dtype=torch.long, device=self.device)
        bpb, idx = torch.sort(bpb, descending=True)
        #print(ax, bpb, idx)
        r = torch.ones_like(sum)
        sr = torch.zeros_like(sum)
        th = (self.B - sum) * k / (self.n - k)
        for i in range(k):
            r = torch.where(sr < th, bpb[:, i], r)
            sr += self.s[ax, idx[:, i]]
        
        #print(r)

        #r = 1.3

        for i in range(k, self.n):
            action = ((self.v[:, i] / self.s[:, i]) < r).double()
            valid = (1 - action) * ((sum + self.s[:, i]) <= self.B)
            sum += valid * self.s[:, i]
            rwd += valid * self.v[:, i]
        
        #print(self.B - sum)
        
        return rwd.mean(0)
