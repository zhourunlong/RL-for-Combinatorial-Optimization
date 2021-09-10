import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable

class SoftmaxTabularAgent():
    def __init__(self, n, lr=1e-3, regular_lambda=1e-4):
        self.n = n
        self.theta = Variable(torch.zeros((n, 2))).cuda()
        self.theta.requires_grad = True
        self.lr = lr
        self.regular_lambda = regular_lambda
    
    def update_n(self, n):
        self.n = n

    def get_action(self, state):
        i, xi = (state[0] * self.n - 0.5).long(), state[1].long()
        params = torch.cat([self.theta[i, xi].view(-1, 1), torch.zeros_like(self.theta[i, xi].view(-1, 1))], dim=-1)
        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)

        action = probs.multinomial(1)
        prob = probs.gather(1, action).view(-1,)
        log_prob = log_probs.gather(1, action).view(-1,)
        entropy = -(probs * log_probs).sum(-1)

        return action.view(-1,), prob, log_prob, entropy, None
        
    def update_param(self, states, rewards, probs, log_probs, entropies, grads_logp):
        states.reverse()
        rewards.reverse()
        probs.reverse()
        log_probs.reverse()
        #entropies.reverse()
        
        rewards = torch.stack(rewards)
        probs = torch.stack(probs)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        
        rewards = rewards.cumsum(0)
        baseline = rewards[-1].mean()
        loss = -(log_probs * (rewards - baseline) + self.regular_lambda * entropies).mean()

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
            F.index_add_(0, idx, (1 - probs.view(-1,)) ** 2)
        F = F.view(n, 2) / (n * bs)
        F[F < 1e-5] = 1e9
        invF = 1 / F

        grads = autograd.grad(outputs=loss, inputs=self.theta, allow_unused=True)[0]
        grads[torch.isnan(grads)] = 0
        self.theta = self.theta - self.lr * invF * grads

        #self.theta.clamp_(-20, 20)

        return rewards[-1].mean().detach().cpu(), loss.detach().cpu()

    def get_accept_prob(self, state):
        with torch.no_grad():
            i, xi = (state[0] * self.n - 0.5).long(), state[1].long()
            return 1 - 1 / (1 + self.theta[i, xi].exp())

class NeuralNetworkAgent():
    def __init__(self, n, lr=1e-3, regular_lambda=1e-4):
        self.n = n
        self.NN = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        ).cuda()
        self.optimizer = torch.optim.Adam(self.NN.parameters(), lr=lr)
        self.regular_lambda = regular_lambda

    def update_n(self, n):
        self.n = n

    def get_action(self, state):
        inputs = torch.cat([state[0].view(-1, 1), state[1].view(-1, 1)], dim=-1)
        params = self.NN(inputs)
        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)

        action = probs.multinomial(1)
        prob = probs.gather(1, action).view(-1,)
        log_prob = log_probs.gather(1, action).view(-1,)
        entropy = -(probs * log_probs).sum(-1)

        return action.view(-1,), prob, log_prob, entropy, None
        
    def update_param(self, states, rewards, probs, log_probs, entropies, grads_logp):
        states.reverse()
        rewards.reverse()
        probs.reverse()
        log_probs.reverse()
        #entropies.reverse()
        
        rewards = torch.stack(rewards)
        probs = torch.stack(probs)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        
        rewards = rewards.cumsum(0)
        baseline = rewards[-1].mean()
        loss = -(log_probs * (rewards - baseline) + self.regular_lambda * entropies).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return rewards[-1].mean().detach().cpu(), loss.detach().cpu()

    def get_accept_prob(self, state):
        with torch.no_grad():
            inputs = torch.cat([state[0].view(-1, 1), state[1].view(-1, 1)],    dim=-1)
            params = self.NN(inputs)
            probs = F.softmax(params, dim=-1)
        return probs[:, 0].view(-1,)

class LogLinearAgent():
    def __init__(self, lr=1e-3, regular_lambda=1e-4, d0=5):
        self.d0 = d0
        self.d = d0 * 2
        self.theta = torch.zeros((self.d, 1), dtype=torch.double, device="cuda")
        self.lr = lr
        self.regular_lambda = regular_lambda
        if regular_lambda != 0:
            print("Regular lambda not used in log-linear policy!")
    
    def update_n(self, n):
        self.n = n

    def get_phi_batch(self, fractions, indicators):
        bs = fractions.shape[0]

        f_axis = torch.ones((bs, self.d0), dtype=torch.double, device="cuda")
        i_axis = torch.cat((torch.ones((bs, 1), dtype=torch.double, device="cuda"), indicators.view(-1, 1)), dim=-1)
        
        fractions = fractions.view(-1,)
        for i in range(1, self.d0):
            f_axis[:, i] = f_axis[:, i - 1] * fractions

        phi = torch.bmm(f_axis.unsqueeze(2), i_axis.unsqueeze(1)).view(bs, -1)
        return phi
    
    def get_phi_all(self):
        f = torch.arange(1, self.n + 1, dtype=torch.double, device="cuda") / self.n
        f = torch.stack((f, f), dim=1).view(-1,)

        x = torch.tensor([0, 1], dtype=torch.double, device="cuda")
        x = x.repeat(self.n, 1).view(-1,)

        phi = self.get_phi_batch(f, x)
        return phi.view(self.n, 2, -1)
    
    def get_action(self, state):
        phi = self.get_phi_batch(state[0], state[1])

        params = torch.matmul(phi, self.theta)
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)

        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)

        action = probs.multinomial(1)
        prob = probs.gather(1, action).view(-1,)
        log_prob = log_probs.gather(1, action).view(-1,)
        #entropy = -(probs * log_probs).sum(-1)

        action = action.double()

        grad_logp = (1 - prob).view(-1, 1) * (1 - 2 * action) * phi

        return action.view(-1,), prob, log_prob, grad_logp
    
    def get_policy(self):
        phi = self.get_phi_all().view(2 * self.n, -1)
        
        params = torch.matmul(phi, self.theta)
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)

        probs = F.softmax(params, dim=-1)

        return probs[:, 0].view(self.n, 2)
        
    def update_param(self, states, rewards, probs, log_probs, grads_logp):
        states.reverse()
        rewards.reverse()
        probs.reverse()
        log_probs.reverse()
        grads_logp.reverse()
        
        rewards = torch.stack(rewards)
        probs = torch.stack(probs)
        log_probs = torch.stack(log_probs)
        grads_logp = torch.stack(grads_logp)
        
        rewards = rewards.cumsum(0)
        baseline = rewards[-1].mean()
        rewards -= baseline
        loss = -(log_probs * rewards).mean()

        # no regularizer!!!!
        rewards.unsqueeze_(1)
        grads = -torch.matmul(rewards, grads_logp).transpose(1, 2).mean(0) / rewards.shape[2]

        F = torch.matmul(grads_logp.view(-1, self.d, 1), grads_logp.view(-1, 1, self.d)).mean(0)
        
        grads = torch.matmul(F.pinverse(), grads)
        norm = torch.norm(grads)
        if norm > 100:
            grads *= 10 / norm.sqrt()

        self.theta -= self.lr * grads

        return baseline.detach().cpu(), loss.detach().cpu()
    
    def get_logits(self, state):
        phi = self.get_phi_batch(state[0], state[1])

        params = torch.matmul(phi, self.theta)

        return params

    def get_accept_prob(self, state):
        params = self.get_logits(state)
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)

        probs = F.softmax(params, dim=-1)

        return probs[:, 0].view(-1,)

if __name__ == "__main__":
    agent = NaiveAgent(3)
    for i in range(3):
        print(agent.get_action([torch.full((3,), (i + 1) / 3, device="cuda"), torch.randint(2, size=(3,), dtype=int, device="cuda")]))
