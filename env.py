import torch

class CSPEnv():
    def __init__(self, bs, type, device, rwd_succ, rwd_fail):
        self.bs = bs
        self.type = type
        self.device = device
        self.rwd_succ = rwd_succ
        self.rwd_fail = rwd_fail
    
    def move_device(self, device):
        self.device = device
        self.probs.to(self.device)
        self.v.to(self.device)
        self.argmax.to(self.device)
        self.active.to(self.device)
    
    def reset_n(self, n):
        self.n = n
        if self.type == "uniform":
            self.probs = 1 / torch.arange(1, self.n + 1, dtype=torch.double, device=self.device)
        else:
            tmp = 1 / torch.arange(2, self.n + 1, dtype=torch.double, device=self.device)
            self.probs = torch.cat((torch.ones((1,), dtype=torch.double, device=self.device), tmp.pow(0.25 + 2 * torch.rand(self.n - 1, dtype=torch.double, device=self.device))))
        self.new_instance()

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
            return self.active * ((1 - action) * raw_reward + action * self.rwd_fail), raw_reward, self.active
        ret = self.active * (1 - action) * raw_reward
        ract = self.active.clone()
        self.active *= action
        return ret, raw_reward, ract
        