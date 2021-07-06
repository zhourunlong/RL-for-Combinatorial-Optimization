import torch

class CSPEnv():
    def __init__(self, n, bs):
        self.n = n
        self.bs = bs
        self.max = torch.full((bs,), n - 1, device="cuda")
    
    def reset(self, reset_perm):
        self.i = 0
        if reset_perm:
            self.v = torch.zeros((self.bs, self.n), device="cuda")
            batch_axis = torch.arange(self.bs, dtype=int, device="cuda")
            for i in range(1, self.n):
                pos = torch.randint(i + 1, (self.bs,), device="cuda")
                self.v[:, i] = self.v[batch_axis, pos]
                self.v[batch_axis, pos] = i
        self.premax = torch.zeros((self.bs,), device="cuda")
        self.active = torch.ones((self.bs,), device="cuda")
    
    def get_state(self):
        self.premax = torch.max(self.premax, self.v[:, self.i])
        return [torch.full((self.bs,), (self.i + 1) / self.n, device="cuda"), torch.eq(self.v[:, self.i], self.premax).int()]
    
    def get_reward(self, action):
        action = action.float()
        raw_reward = 2 * torch.eq(self.v[:, self.i], self.max).float() - 1
        self.i += 1
        if self.i == self.n:
            return self.active * raw_reward
        ret = (1 - action) * self.active * raw_reward
        self.active *= action
        return ret
    
    def print_v(self):
        print(self.v)

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
        