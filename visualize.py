import torch
import torch.nn.functional as F
from agent import *
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from calculate_opt_policy import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", default=None, type=str)
    return parser.parse_args()

def plot_prob_fig(agent, env, pic_dir, device):
    states = torch.stack((torch.arange(1, agent.n + 1, dtype=torch.double, device=device) / agent.n, torch.ones((agent.n,), dtype=torch.double, device=device)), dim=1)
    max_acc = agent.get_accept_prob(states).cpu().numpy()

    states = torch.stack((torch.arange(1, agent.n + 1, dtype=torch.double, device=device) / agent.n, torch.zeros((agent.n,), dtype=torch.double, device=device)), dim=1)
    non_max_acc = agent.get_accept_prob(states).cpu().numpy()

    fig, ax = plt.subplots(figsize=(20, 20))

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    x = np.array([_ / agent.n for _ in range(agent.n + 1)])
    ax.plot(x, np.concatenate((np.zeros(1,), np.array(max_acc))), label="P[Accept|SufMax]")
    ax.plot(x, np.concatenate((np.zeros(1,), np.array(non_max_acc))), label="P[Accept|NotSufMax]")

    idx = opt_tabular(env.probs.cpu().numpy())
    y = np.zeros_like(x)
    for i in idx:
        y[i] = 1
    ax.plot(x, y, label="Optimal")

    ax.set_title("Plot of acceptance probability", fontsize=40)
    ax.set_xlabel("Time", fontsize=40)
    ax.set_ylabel("Prob", fontsize=40)
    ax.legend(loc="best", fontsize=40)

    plt.savefig(pic_dir)
    plt.close()

def plot_rl_fig(val1, label1, val2, label2, pic_dir, curve_buffer_size, len_avail):
    fig, ax1 = plt.subplots(figsize=(40, 20))

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    
    ax2 = ax1.twinx()

    x = np.array([_ * curve_buffer_size for _ in range(len_avail)])
    line1, = ax1.plot(x, val1[:len_avail], "g-", label=label1)
    line2, = ax2.plot(x, val2[:len_avail], "b--", label=label2)

    plt.title("Plot of training curve", fontsize=40)
    ax1.set_xlabel("Episode", fontsize=40)
    plt.legend((line1, line2), (label1, label2), loc="best", fontsize=40)

    plt.savefig(pic_dir)
    plt.close()

if __name__ == "__main__":
    args = get_args()
    package = torch.load(args.load_path)
    env = package["env"]
    agent = package["agent"]
    #plot_prob_fig(agent, "visualize.jpg")
    #plot_rl_fig([0.5, 0.6, 0.7], [-0.1, -0.2, -0.3], "curve.jpg")
    #print(agent.theta.view(-1,))

    idx = opt_tabular(env.probs)
    (success, x) = opt_loglinear(env.n, agent.d0, 5, (idx[-1] - 0.5) / env.n)
    assert success

    x = torch.tensor(x).double().cuda()
    print("agent norm:", torch.norm(agent.theta).item())
    plot_prob_fig(agent, env, "visualize_agent.jpg")

    agent.theta = x.view_as(agent.theta)
    print("optimal norm:", torch.norm(x).item())
    plot_prob_fig(agent, env, "visualize_optimal.jpg")

    states_1 = [torch.arange(1, agent.n + 1, dtype=torch.double, device="cuda") / agent.n, torch.ones((agent.n,), dtype=torch.double, device="cuda")]
    states_0 = [torch.arange(1, agent.n + 1, dtype=torch.double, device="cuda") / agent.n, torch.zeros((agent.n,), dtype=torch.double, device="cuda")]
    logits_1 = agent.get_logits(states_1).view(-1,).cpu()
    logits_0 = agent.get_logits(states_0).view(-1,).cpu()

    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.array([(_ + 1) / agent.n for _ in range(agent.n)])
    ax.plot(x, np.array(logits_1), label="Logits_1")
    ax.plot(x, np.array(logits_0), label="Logits_0")
    plt.savefig("logits.jpg")
    plt.close()
