import torch

class CSPEnv():
    def __init__(self, bs, type):
        self.bs = bs
        self.type = type
    
    def reset(self, reset_perm, reset_n = None):
        self.i = 0
        if reset_n is not None:
            self.n = reset_n
            if self.type == "uniform":
                self.probs = 1 / torch.arange(1, self.n + 1, dtype=torch.double, device="cuda")
            else:
                tmp = 1 / torch.arange(2, self.n + 1, dtype=torch.double, device="cuda")
                self.probs = torch.cat((torch.ones((1,), dtype=torch.double, device="cuda"), tmp.pow(2 * torch.rand(self.n - 1, dtype=torch.double, device="cuda"))))

        if reset_perm:
            self.v = self.probs.repeat(self.bs, 1).bernoulli()
            self.argmax = torch.argmax(self.v + torch.arange(self.n, dtype=torch.double, device="cuda") * 1e-5, 1)
        
        self.active = torch.ones((self.bs,), dtype=torch.double, device="cuda")

        print(self.v)
    
    def get_state(self):
        return [torch.full((self.bs,), (self.i + 1) / self.n, dtype=torch.double, device="cuda"), self.v[:, self.i].double()]
    
    def get_reward(self, action):
        raw_reward = 2 * (self.argmax == self.i).double() - 1
        self.i += 1
        ret0 = self.active * raw_reward
        if self.i == self.n:
            return self.active, self.active * ((1 - action) * raw_reward - action), ret0
        ret = (1 - action) * ret0
        ract = self.active.clone()
        self.active *= action
        return ract, ret, ret0

if __name__ == "__main__":
    env = CSPEnv(5, 3)
    
    for _ in range(3):
        print("----- trial", _, "-----")
        env.reset()
        env.print_v()

        for i in range(5):
            action = (torch.rand((3,)) < 0.7).double().cuda()
            state = env.get_state()
            print(state[0], state[1], action.int(), env.get_reward(action).int())
        
        print("-------------------")
        