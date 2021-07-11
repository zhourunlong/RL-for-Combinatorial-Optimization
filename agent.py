import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable

class NaiveAgent():
    def __init__(self, n, lr=1e-3):
        self.n = n
        self.theta = Variable(torch.zeros((n, 2))).cuda()
        self.theta.requires_grad = True
        self.lr = lr

    def get_action(self, state):
        i, xi = (state[0] * self.n - 0.5).long(), state[1].long()
        probs = (1 - 1 / (1 + self.theta[i, xi].exp())).view(-1, 1)
        probs = torch.cat([probs, 1 - probs], dim=-1)

        action = probs.multinomial(1)
        log_prob = probs.gather(1, action).log().view(-1,)
        entropy = -(probs * probs.log()).sum(-1)

        return action.view(-1,), log_prob, entropy
        
    def update_param(self, states, rewards, log_probs, entropies):
        rewards.reverse()
        log_probs.reverse()
        
        rewards = torch.stack(rewards)
        log_probs = torch.stack(log_probs)

        # https://zhuanlan.zhihu.com/p/78684058
        '''
        sum_log_probs = log_probs.sum(0).mean()
        grad_slp = autograd.grad(outputs=sum_log_probs, inputs=self.theta, retain_graph=True, allow_unused=True)[0]
        grad_norm_sq = (grad_slp * grad_slp).sum()
        baseline =  / grad_norm_sq
        '''
        
        rewards = rewards.cumsum(0)
        baseline = rewards[-1].mean()
        loss = -(log_probs * (rewards - baseline)).sum(0).mean()

        loss = loss / len(rewards)

        grads = autograd.grad(outputs=loss, inputs=self.theta, allow_unused=True)[0]
        self.theta = self.theta - self.lr * grads

        return rewards[-1].mean().detach().cpu(), loss.detach().cpu()

    def get_accept_prob(self, state):
        i, xi = (state[0] * self.n - 0.5).long(), state[1].long()
        return 1 - 1 / (1 + self.theta[i, xi].exp())

class NPGAgent():
    def __init__(self, n, lr=1e-3):
        self.n = n
        self.theta = Variable(torch.zeros((n, 2))).cuda()
        self.theta.requires_grad = True
        self.lr = lr

    def get_action(self, state):
        i, xi = (state[0] * self.n - 0.5).long(), state[1].long()
        probs = (1 / (1 + self.theta[i, xi].exp())).view(-1, 1)
        probs = torch.cat([1 - probs, probs], dim=-1)

        action = probs.multinomial(1)
        log_prob = probs.gather(1, action).log().view(-1,)
        entropy = -(probs * probs.log()).sum(-1)

        return action.view(-1,), log_prob, entropy
        
    def update_param(self, states, rewards, log_probs, entropies):
        states.reverse()
        rewards.reverse()
        log_probs.reverse()
        
        rewards = torch.stack(rewards)
        log_probs = torch.stack(log_probs)
        
        rewards = rewards.cumsum(0)
        loss = -(log_probs * rewards).sum(0).mean()

        n, bs = log_probs.shape
        F = torch.zeros(n * 2, device="cuda")

        __i, __xi = [], []
        for i in range(n):
            __i.append(states[i][0])
            __xi.append(states[i][1])
        _i, _xi = torch.cat(__i, dim=-1), torch.cat(__xi, dim=-1)
        _i, _xi = (_i * self.n - 0.5).long(), _xi.long()
        idx = 2 * _i + _xi

        with torch.no_grad():
            F.index_add_(0, idx, (1 - log_probs.view(-1,).exp()) ** 2)
        F = F.view(n, 2) / (n * bs)
        F[F < 1e-5] = 1e9
        invF = 1 / F

        loss = loss / len(rewards)

        grads = autograd.grad(outputs=loss, inputs=self.theta, allow_unused=True)[0]
        grads[torch.isnan(grads)] = 0
        self.theta = self.theta - self.lr * invF * grads

        return rewards[-1].mean().detach().cpu(), loss.detach().cpu()

    def get_accept_prob(self, state):
        i, xi = (state[0] * self.n - 0.5).long(), state[1].long()
        return 1 - 1 / (1 + self.theta[i, xi].exp())

if __name__ == "__main__":
    agent = NaiveAgent(3)
    for i in range(3):
        print(agent.get_action([torch.full((3,), (i + 1) / 3, device="cuda"), torch.randint(2, size=(3,), dtype=int, device="cuda")]))