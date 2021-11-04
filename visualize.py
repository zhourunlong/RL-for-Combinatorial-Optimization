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
    parser.add_argument("--load-path", default=None, type=str)
    return parser.parse_args()

def plot_prob_fig(agent, env, pic_dir, device):
    n = 1000
    f = torch.arange(0, n + 1, dtype=torch.double, device=device) / n
    states = torch.stack((f, torch.ones_like(f)), dim=1)
    max_acc = agent.get_accept_prob(states).cpu().numpy()
    states = torch.stack((f, torch.zeros_like(f)), dim=1)
    non_max_acc = agent.get_accept_prob(states).cpu().numpy()

    fig, ax = plt.subplots(figsize=(20, 20))

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    x = np.arange(0, n + 1) / n
    ax.plot(x, max_acc, label="P[Accept|PrefMax]")
    ax.plot(x, non_max_acc, label="P[Accept|NotPrefMax]")

    idx = env.get_opt_policy(return_idx=True)
    y = np.zeros_like(x)
    th = np.min(idx)
    i1 = n * (th - 1) // env.n
    i2 = (n * th + env.n - 1) // env.n
    y[i1:i2] = np.arange(0, 1, 1 / (i2 - i1))
    y[i2:] = 1
    ax.plot(x, y, label="Optimal")

    ax.set_title("Plot of Policy", fontsize=40)
    ax.set_xlabel("Time", fontsize=30)
    ax.set_ylabel("Prob", fontsize=30)
    ax.legend(loc="best", fontsize=30)

    plt.savefig(pic_dir)
    plt.close()

def plot_rl_fig(x, labelx, y1, label1, y2, label2, pic_dir):
    matplotlib.rc('font', size=30)
    fig, ax1 = plt.subplots(figsize=(40, 20))
    ax1.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True, axis="x")
    
    ax2 = ax1.twinx()
    
    line1, = ax1.plot(x, y1, "g-")
    ax1.set_ylabel(label1)
    ax1.tick_params(axis='y')

    line2, = ax2.plot(x, y2, "b--")
    ax2.set_ylabel(label2)
    ax2.tick_params(axis='y')

    plt.title("Plot of Training Curve", fontsize=40)
    ax1.set_xlabel(labelx)
    plt.legend((line1, line2), (label1, label2), loc="best")

    plt.savefig(pic_dir)
    plt.close()
