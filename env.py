import torch

class CSPEnv():
    def __init__(self, n, bs, type):
        self.n = n
        self.bs = bs
        self.type = type
        self.reset(True, n)
    
    def reset(self, reset_perm, reset_n = None):
        self.i = 0
        if reset_n is not None:
            self.n = reset_n
            if self.type == "uniform":
                self.probs = 1 / torch.arange(1, self.n + 1, dtype=torch.float32, device="cuda")
            else:
                self.probs = torch.cat((torch.ones((1,), device="cuda"), torch.rand(self.n - 1, device="cuda")))

        if reset_perm:
            self.v = self.probs.repeat(self.bs, 1).bernoulli()
            self.argmax = torch.argmax(self.v + torch.arange(self.n, dtype=torch.float32, device="cuda") * 1e-5, 1)
        
        self.active = torch.ones((self.bs,), device="cuda")
    
    def get_state(self):
        return [torch.full((self.bs,), (self.i + 1) / self.n, device="cuda"), self.v[:, self.i].float()]
    
    def get_reward(self, action):
        action = action.float()
        raw_reward = 2 * (self.argmax == self.i).float() - 1
        self.i += 1
        if self.i == self.n:
            return self.active * ((1 - action) * raw_reward - action)
        ret = (1 - action) * self.active * raw_reward
        self.active *= action
        return ret

if __name__ == "__main__":
    env = CSPEnv(5, 3)
    
    for _ in range(3):
        print("----- trial", _, "-----")
        env.reset()
        env.print_v()

        for i in range(5):
            action = (torch.rand((3,)) < 0.7).float().cuda()
            state = env.get_state()
            print(state[0], state[1], action.int(), env.get_reward(action).int())
        
        print("-------------------")
        