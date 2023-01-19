from .base_agent import LogLinearAgent
import torch
import torch.nn.functional as F


class ADWAgent(LogLinearAgent):
    def __init__(self, device, d0, lr, L, G, U, **kwargs):
        self.d0 = int(d0)
        self.d = self.d0 ** 3
        self.m = 10
        super().__init__(device, lr, L, G, U)

    def move_device(self, device):
        self.device = device
        self.theta.to(self.device)

    def set_curriculum_params(self, param):
        self.n = param[0]
        self.m = param[1]

    def clear_params(self):
        self.theta = torch.zeros_like(self.theta)

# new state: i/n & v_i/remaining_budget_i
    def get_phi_batch(self, states):
        # self.m phis?
        bs = states.shape[0]
#        self.m = self.m
        return_value = torch.empty(
            bs, self.d, 1, dtype=torch.double, device=self.device)
        for j in range(0, self.m):
            ax = torch.zeros((bs, 3, self.d0),
                             dtype=torch.double, device=self.device)
            ax[:, :, 0] = 1
            for i in range(1, self.d0):
                ax[:, :, i] = ax[:, :, i - 1] * torch.cat(
                    (states[:, 0:1], states[:, j * 2 + 1:j * 2 + 3]), dim=1)
            phi = torch.bmm(ax[:, 0, :].unsqueeze(
                2), ax[:, 1, :].unsqueeze(1)).view(bs, -1, 1)
            phi = torch.bmm(phi, ax[:, 2, :].unsqueeze(1)).view(bs, -1)
            return_value = torch.cat((return_value, phi.unsqueeze(2)), dim=2)
        return return_value[:, :, 1:self.m+1]

    def get_action(self, states):
        params = self.get_logits(states)
 #       print("pa", params.shape)
        params = torch.cat([params, torch.zeros_like(params[:, 0:1])], dim=-1)

        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)
        entropy = -(probs * log_probs).sum(-1)

        action = probs.multinomial(1)
        return action.double().view(-1,), entropy

    def query_sa(self, states, actions):
        phi = self.get_phi_batch(states)

        params = self.get_logits(states)
        params = torch.cat([params, torch.zeros_like(params[:, 0:1])], dim=-1)

        probs = F.softmax(params, dim=-1)
        log_probs = F.log_softmax(params, dim=-1)
        actions = actions.long().view(states.shape[0], 1)

#        print("act ", actions)
#        print("pbs ", probs)
        range = torch.arange(
            0, phi.shape[0], dtype=torch.long, device=self.device)
        range_2 = torch.arange(
            0, phi.shape[2], dtype=torch.long, device=self.device)
        prob = probs.gather(1, actions).view(-1,)
#        print("prob", prob)
#        print("pb ", prob)
#        print("shape actions", actions.shape)
        single_action = actions.squeeze(1)

#        print("shape phi", phi[range, :, single_action])
#        print("prob shape", (1 - prob).view(-1, 1).shape)
        log_prob = log_probs.gather(1, actions).view(-1,)
        grad_logp = torch.empty(
            phi.shape[0], phi.shape[1], dtype=torch.double, device=self.device)
        phi = torch.cat((phi, torch.zeros_like(phi[:, :, 0:1])), dim=-1)
        grad_logp[range] = (1 - prob[range]).view(-1, 1) * \
            (1 - 2 * (actions[range] == self.m)) * phi[range, :, single_action]
        return log_prob, grad_logp

    def get_logits(self, states):
        phi = self.get_phi_batch(states)
        return_value = phi[:, :, 0] @ self.theta
        for i in range(1, phi.shape[2]):
            return_value = torch.cat(
                (return_value, phi[:, :, i] @ self.theta), dim=1)
        return return_value

    def get_accept_prob(self, states):
        params = self.get_logits(states)
        return torch.sigmoid(params).view(-1,)
