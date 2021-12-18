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

def plot_rl_fig(x, labelx, y1, label1, y2, label2, newax, pic_dir):
    matplotlib.rc('font', size=30)
    fig, ax1 = plt.subplots(figsize=(40, 20))
    ax1.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True, axis="x")
    
    if newax:
        ax2 = ax1.twinx()
    else:
        ax2 = ax1
    
    line2, = ax2.plot(x, y2, "b--")
    ax2.set_ylabel(label2)
    ax2.tick_params(axis='y')

    line1, = ax1.plot(x, y1, "g-")
    ax1.set_ylabel(label1)
    ax1.tick_params(axis='y')

    plt.title("Plot of Training Curve", fontsize=40)
    ax1.set_xlabel(labelx)
    plt.legend((line1, line2), (label1, label2), loc="best")

    plt.savefig(pic_dir)
    plt.close()

