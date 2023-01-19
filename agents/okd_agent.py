from .base_agent import LogLinearAgent
import torch
import torch.nn.functional as F


class OKDAgent(LogLinearAgent):
    def __init__(self, device, d0, lr, L, G, U, **kwargs):
        self.d0 = int(d0)
        self.d = self.d0 ** 5
        super().__init__(device, lr, L, G, U)

    def move_device(self, device):
        self.device = device
        self.theta.to(self.device)

    def set_curriculum_params(self, param):
        self.n = param[0]

    def clear_params(self):
        self.theta = torch.zeros_like(self.theta)

    def get_phi_batch(self, states):
        bs = states.shape[0]

        ax = torch.zeros((bs, 5, self.d0), dtype=torch.double,
                         device=self.device)
        ax[:, :, 0] = 1
        for i in range(1, self.d0):
            ax[:, :, i] = ax[:, :, i - 1] * states

        t1 = torch.bmm(ax[:, 0, :].unsqueeze(
            2), ax[:, 1, :].unsqueeze(1)).view(bs, -1, 1)
        t1 = torch.bmm(t1, ax[:, 2, :].unsqueeze(1)).view(bs, -1, 1)
        t2 = torch.bmm(ax[:, 3, :].unsqueeze(
            2), ax[:, 4, :].unsqueeze(1)).view(bs, 1, -1)
        phi = torch.bmm(t1, t2).view(bs, -1)
        return phi

    def get_action(self, states):
        params = self.get_logits(states)
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)
        # print(params.shape)

        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)
        entropy = -(probs * log_probs).sum(-1)

        action = probs.multinomial(1)
        return action.double().view(-1,), entropy

    def query_sa(self, states, actions):
        phi = self.get_phi_batch(states)

        params = phi @ self.theta
        params = torch.cat([params, torch.zeros_like(params)], dim=-1)

        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)
        actions = actions.long().view(states.shape[0], 1)

        prob = probs.gather(1, actions).view(-1,)
#        print("prob", prob)
        log_prob = log_probs.gather(1, actions).view(-1,)
        # print("shape actions", actions.shape)
        # print("shape phi", phi.shape)
        # print("prob shape", (1 - prob).view(-1, 1).shape)
        grad_logp = (1 - prob).view(-1, 1) * (1 - 2 * actions) * phi
        # print("shape grad", grad_logp.shape)
        return log_prob, grad_logp

    def get_logits(self, states):
        phi = self.get_phi_batch(states)
        return phi @ self.theta

    def get_accept_prob(self, states):
        params = self.get_logits(states)
        return torch.sigmoid(params).view(-1,)
