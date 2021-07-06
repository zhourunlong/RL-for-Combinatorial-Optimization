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

if __name__ == "__main__":
    args = get_args()

    agent = torch.load(args.model_dir)
    #print(agent.theta)

    max_acc, non_max_acc = [], []
    for i in range(agent.n):
        max_acc.append(F.softmax(agent.theta[i, 1], dim=0)[0])
        non_max_acc.append(F.softmax(agent.theta[i, 0], dim=0)[0])

    #print(max_acc)
    #print(non_max_acc)

    fig, ax = plt.subplots(figsize=(10, 10))

    x = np.array([(_ + 1) / agent.n for _ in range(agent.n)])
    ax.plot(x, np.array(max_acc), label="Agent")
    #ax.plot(x, np.array(non_max_acc), label="not prefix max")

    x2 = np.array([1 / agent.n, np.exp(-1), np.exp(-1), 1])
    y2 = np.array([0, 0, 1, 1])
    ax.plot(x2, y2, label="Optimal")

    ax.set_title("Plot of acceptance probability")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pr[accept]")
    ax.legend(loc="best")

    plt.savefig('visualize.png')
    #plt.show()
