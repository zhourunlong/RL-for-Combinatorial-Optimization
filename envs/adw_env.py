from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from .base_env import BaseEnv
import torch
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')


class ADWEnv(BaseEnv):
    def __init__(self, device, distr_type, distr_granularity, batch_size, **kwargs):
        super().__init__(device, distr_type, batch_size)
        self.gran = int(distr_granularity)

    def move_device(self, device):
        self.device = device
        self.Fv.to(self.device)

# params
# n-> n ad slots
# m-> m advertisers
# ! B-> array[1..m]->budgets of advertisers
    def set_curriculum_params(self, param):
        self.plot_states = None
        [self.n, self.m, self.V] = param
        self.horizon = self.n
        self.bs = self.bs_per_horizon * (self.horizon + 1)
        if self.type == "uniform":
            self.Fv = torch.ones(
                (self.gran,), dtype=torch.double, device=self.device)
        elif self.type == "random":
            self.Fv = torch.rand(
                (self.gran,), dtype=torch.double, device=self.device)
        else:
            self.Fv = torch.rand(
                (self.gran,), dtype=torch.double, device=self.device)
        self.new_instance()

#        self.calc_ref_r()
        self.clean()

    @property
    def curriculum_params(self):
        return [self.n, self.m, self.V]

# action_size -> m advertisers, or reject all
    @property
    def action_size(self):
        return self.m + 1

    def sample_distr(self, F):
        if (self.type != "special"):
            interval = torch.multinomial(
                F, self.bs * self.n * (self.m), replacement=True)
            sample = interval.double() + torch.rand(self.bs * self.n * (self.m),
                                                    dtype=torch.double, device=self.device)
            return sample.view(self.bs, self.n, self.m) / self.gran
        else:
            sample = torch.rand(self.bs * self.n * self.m,
                                dtype=torch.double, device=self.device)
            sample = 0.4 * (sample < 0.9) + \
                (4 * (sample - 0.9) + 0.6) * (sample >= 0.9)
            return sample.view(self.bs, self.n, self.m)
# sum_v -> value each advertiser spent
# v -> value i,j
# total_revenue -> total_revenue

# B -> all 1
    def new_instance(self):
        self.i = 0
        self.total_revenue = torch.zeros(
            (self.bs,), dtype=torch.double, device=self.device)
        self.v = torch.cat((self.sample_distr(self.Fv), torch.zeros(
            self.bs, self.n, 1, device=self.device, dtype=torch.double)), 2)
        self.sum_v = torch.zeros(
            (self.bs, (self.m + 1)), dtype=torch.double, device=self.device)
        self.active = torch.ones(
            (self.bs), dtype=torch.double, device=self.device)
# TODO: give B a dist
        self.B = torch.ones_like(self.sum_v)

# new state: i/n & v_i/remaining_budget_i
    def get_state(self):
        state = [torch.full((self.bs,), (self.i + 1) / self.n,
                            dtype=torch.double, device=self.device)]
        for j in range(0, self.m):
            state.append(self.v[:, self.i, j])
            state.append(self.B[:, j] - self.sum_v[:, j])
        return torch.stack(state, dim=1)
        # state = list(self.v[:, self.i, ],
        #              self.v[:, self.i, ].shape[0], dim=1)
        # state.append(torch.full((self.bs,), (self.i + 1) / self.n,
        #              dtype=torch.double, device=self.device))
        # state.append(self.sum_v)
        # return torch.stack(state, dim=1)

        # get_reference_action
        # action:
        # 0: reject every advertiser
        # 1-m: {advertiser_action}
    def get_reference_action(self):
        return torch.argmax((self.v[:, self.i, :] * (math.e ** (self.sum_v / self.B))), 1).double()
        # return torch.argmax(1 / (0.5 - self.v[:, self.i, :]) * ((self.B - self.sum_v) > self.v[:, self.i, :]), 1).double()
        # return torch.argmax((self.v[:, self.i, :] / (self.B - self.sum_v)) * (self.v[:, self.i, :] <= (self.B - self.sum_v)), 1).double()
        #    return (self.v[:, self.i] < self.r * self.s[:, self.i]).double()

        # Get Reward
    def get_reward(self, action):
        #        pickable = []
        #        for j in range(self.m):
        #            pickable.append(self.v[:, self.i, j] + self.sum_v[j]  < self.B[j])
        # TODO: reduce valid bs * m -> bs get valid bs_range & action pairs
        bs_range = torch.arange(
            0, self.bs, device=self.device, dtype=torch.long)
        action = action.long()
        valid = torch.zeros(self.bs, self.m + 1,
                            device=self.device, dtype=torch.double)
        valid[bs_range, action] = 1 * self.active[bs_range] * \
            (self.v[bs_range, self.i, action] +
             self.sum_v[bs_range, action] < self.B[bs_range, action])
        # sum_v_1 = self.sum_v + self.v[:, self.i, :] * self.active[bs_range] * \
        #     (self.v[bs_range, self.i, action] +
        #      self.sum_v[bs_range, action] < self.B[bs_range, action])
        addValue = self.v[:, self.i, :] * valid
        self.sum_v += addValue
        # print(torch.norm(sum_v_1 - self.sum_v))
        addValue = addValue[bs_range, action]
        self.total_revenue += addValue
        # self.total_revenue += self.sum_v[bs_range, action]
        # self.sum_v = torch.cat((self.sum_v, torch.zeros(
        #     self.bs, 1, dtype=torch.double, device=self.device)), 1)
        # self.sum_v[range(0, self.bs), action.long(
        # )] += 0 if action[range(0, self.bs)] == self.m else (self.v[range(0, self.bs), self.i, action.long()] * self.active * (self.v[rang    e(0, self.bs), self.i, action.long()] + self.sum_v[range(0, self.bs), action.long()] < self.B[range(0, self.bs), action.long()]))
        # self.sum_v = self.sum_v[:, 0:self.m]
        # self.total_revenue += self.sum_v[range(0, self.bs), action.long()]
        # self.sum_v += self.v[]
        # pytorch 高维数组批量寻址
        # for j in range(0, valid.shape[0]):
        #     cur_action = int(action[j].item())
        #     if (cur_action != self.m):
        # self.v[:, ]
        # valid[j] = valid[j] * ((self.v[j, self.i, cur_action] +self.B
        #                        self.sum_v[j, cur_action]) < self.B[j, cur_action])
        # self.sum_v[j, cur_action] += self.v[j, self.i, cur_action]
        # self.total_revenue[j] += self.v[j, self.i, cur_action]
        # action = m -> not pick anything
        # die = self.active * (1 - action) * (1 - pickable.double())
#        self.sum_v[win] = self.V

        win = self.total_revenue >= self.V
        self.total_revenue[win] = self.V
        rwd = self.active * win
#        rwd = addValue
        ract = self.active.clone()
        self.active *= (1 - win.double())
#       self.active *= (1 - win.double())  # * (1 - die.double())
        self.i += 1
# env.bs        print(torch.max(rwd))
        return rwd, ract

    def clean(self):
        del self.v, self.sum_v, self.plot_states
        self.plot_states = None

    def get_plot_states(self):
        print("skip")
        # if self.plot_states is not None:
        #     return self.plot_states

        # x = torch.linspace(0.02, 1, 50, device=self.device)
        # f = torch.linspace(0.1, 0.9, 9, device=self.device)
        # r = torch.linspace(0.1, 0.9, 6, device=self.device)
        # v, s, f, r, q = torch.meshgrid(
        #     x, x, f, r, torch.zeros((1,), device=self.device))
        # self.plot_states = torch.stack(
        #     (v.reshape(-1,), s.reshape(-1,), f.reshape(-1,), r.reshape(-1,), q.reshape(-1,)), dim=1)

        # return self.plot_states

        # two demensions: value & value / budget for average advertisers
    def plot_prob_figure(self, agent, pic_dir):
        print("skip")
        # fig = plt.figure(figsize=(22, 25))
        # color_map = "viridis"

        # acc = agent.get_accept_prob(self.get_plot_states()).view(
        #     50, 50, 9, 6, 1).cpu().numpy()
        # x = np.linspace(0.02, 1, 50)
        # X, Y = np.meshgrid(x, x, indexing="ij")

        # for t in range(9):
        #     ax = fig.add_subplot(3, 3, t + 1, projection='3d')
        #     ax.set_title("i/n = 0.%d" % (t + 1))

        #     for i in range(6):
        #         z = i / 6
        #         ax.contourf(X, Y, z + 0.02 / 6 * acc[:, :, t, i, 0], zdir='z', levels=50,
        #                     cmap=color_map, norm=matplotlib.colors.Normalize(vmin=z, vmax=z + 0.02 / 6))

        #     ax.set_xlim3d(0, 1)
        #     ax.set_xlabel("v")
        #     ax.set_ylim3d(0, 1)
        #     ax.set_ylabel("s")
        #     ax.set_zlim3d(-0.01, 1.01 - 1 / 6)
        #     ax.set_zlabel("sum/B")
        #     ax.invert_zaxis()
        #     ax.view_init(-170, 60)

        # fig.subplots_adjust(wspace=0, hspace=0, right=0.9)
        # position = fig.add_axes([0.92, 0.4, 0.015, 0.2])
        # cb = fig.colorbar(cm.ScalarMappable(
        #     norm=matplotlib.colors.Normalize(0, 1), cmap=color_map), cax=position)

        # plt.savefig(pic_dir, bbox_inches="tight")
        # plt.close()
