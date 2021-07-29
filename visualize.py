import torch
import torch.nn.functional as F
from agent import *
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None, type=str)
    return parser.parse_args()

def plot_prob_fig(agent, pic_dir):
    states_1 = [torch.arange(1, agent.n + 1, device="cuda").float() / agent.n, torch.ones((agent.n,), device="cuda")]
    states_0 = [torch.arange(1, agent.n + 1, device="cuda").float() / agent.n, torch.zeros((agent.n,), device="cuda")]
    with torch.no_grad():
        max_acc = agent.get_accept_prob(states_1).cpu().numpy()
        non_max_acc = agent.get_accept_prob(states_0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))

    x = np.array([(_ + 1) / agent.n for _ in range(agent.n)])
    ax.plot(x, np.array(max_acc), label="Agent")
    ax.plot(x, np.array(non_max_acc), label="Agent (not prefix max)")

    x2 = np.array([0, np.exp(-1), np.exp(-1), 1])
    y2 = np.array([0, 0, 1, 1])
    ax.plot(x2, y2, label="Optimal")

    ax.set_title("Plot of acceptance probability")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pr[accept]")
    ax.legend(loc="best")

    plt.savefig(pic_dir)
    plt.close()

def plot_rl_fig(reward, loss, pic_dir, curve_buffer_size, len_avail):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = ax1.twinx()

    x = np.array([_ * curve_buffer_size for _ in range(len_avail)])
    line1, = ax1.plot(x, reward[:len_avail], "g-", label="Reward")
    line2, = ax2.plot(x, loss[:len_avail], "b--", label="Loss")

    plt.title("Plot of training curve")
    ax1.set_xlabel("Episode")
    plt.legend((line1, line2), ("Reward", "Loss"), loc="best")

    plt.savefig(pic_dir)
    plt.close()

def print_theta(agent):
    print(agent.theta)

if __name__ == "__main__":
    args = get_args()
    agent = torch.load(args.model_dir)
    #plot_prob_fig(agent, "visualize.jpg")
    #plot_rl_fig([0.5, 0.6, 0.7], [-0.1, -0.2, -0.3], "curve.jpg")
    print_theta(agent)
