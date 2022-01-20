from curses import window
from hashlib import new
import torch as th
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from collections import defaultdict

exp_dirs = {                              # sample      init        reg?
    "t-0/0/0":  "Exp-20220114-121504-SP", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  "Exp-20220114-121550-SP", # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20220114-165316-SP", # pi^t	    pi^0        0
    "t/0/r":    "Exp-20220114-121419-SP", # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20220114-165201-SP", # pi^0	    pi^0        0
    "0/0/r":    "Exp-20220114-165356-SP", # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20220114-120904-SP", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  "Exp-20220114-121237-SP", # pi^t	    pi^0-pi^t   reg
}

colors = {
    "t-0/0/0":  [216, 30, 54],
    "t-0/0/r":  [254, 151, 0],
    "t/0/0":    [0, 178, 238],
    "t/0/r":    [160, 32, 240],
    "0/0/0":    [180, 180, 180],
    "0/0/r":    [139, 101, 8],
    "t/0-t/0":  [0, 100, 0],
    "t/0-t/r":  [204, 153, 255],
    "ref":      [50, 240, 80],
}

colors = {key: np.array(val, np.float) / 255. for key, val in colors.items()}

lines = {
    "t-0/0/0":  "-",
    "t-0/0/r":  "--",
    "t/0/0":    "-",
    "t/0/r":    "--",
    "0/0/0":    "-",
    "0/0/r":    "--",
    "t/0-t/0":  "-",
    "t/0-t/r":  "--",
    "ref":      "-",
}

plots = [
    {
        "exps": ["t/0/0", "t/0/r", "t/0-t/0"],
        "vals": ["reward"],
    },
    {
        "exps": ["t-0/0/0", "0/0/0"],
        "vals": ["reward", "log(Kappa)", "err_t"],
    },
]

alpha = 0.2
scale = 50.
confidence = 0.95
font_size = 26
legend_font_size = 32
anchor = (0.5, 1.04)
linewidth = 3
window_size=100

def read_data(dir):
    log_dir = os.path.join(dir, "logdata", "final")
    files = os.listdir(log_dir)
    files.sort()
    v = defaultdict(list)
    for file in files:
        package = th.load(os.path.join(log_dir, file), map_location="cpu")
        for key, val in package.items():
            key = key[6:] # remove prefix (final)
            if key != "episode":
                v[key].append(val)
    ret = {}
    for key, val in v.items():
        ret[key] = th.cat(val).numpy()
    return ret

def get_window(data, ws):
    n = data.shape[0]
    ret = np.zeros((n, 2 * ws + 1))
    ret[:, ws] = data
    for i in range(ws):
        ret[i+1:, ws+i+1] = data[:-(i+1)]
        ret[:-(i+1), ws-i-1] = data[i+1:]
    return ret

def get_percentile_from_window(data, ws, percentile):
    n = data.shape[0]
    ret = np.zeros((n,))
    ret[ws:n-ws] = np.percentile(data[ws:n-ws], percentile, axis=1)
    for i in range(ws):
        ret[i] = np.percentile(data[i,:ws+i], percentile)
        ret[n-1-i] = np.percentile(data[n-1-i,ws-i:], percentile)
    return ret

def smooth_from_window(data, ws):
    n = data.shape[0]
    ret = data.sum(1)
    ret[ws:n-ws] /= 2 * ws + 1
    for i in range(ws):
        ret[i] /= ws + 1 + i
        ret[n-1-i] /= ws + 1 + i
    return ret

def plot_line(ax, x, y, window_size, alpha, color, linewidth, linestyle):
    yw = get_window(y, window_size)

    low = get_percentile_from_window(yw, window_size, 25)
    high = get_percentile_from_window(yw, window_size, 75)
    yy = smooth_from_window(yw, window_size)

    ax.fill_between(x, low, high, alpha=alpha, color=color, linewidth=0)
    ax.plot(x, yy, color=color, label=exp, linewidth=linewidth, linestyle=linestyle)

    return yy.min(), yy.max()

matplotlib.rc('font', size=font_size)
figure = plt.figure(figsize=(32, 22))
tot_sub_plot = sum(len(plot["vals"]) for plot in plots)
idx = 0

used_exps = []
for plot in plots:
    data = {}
    for exp in plot["exps"]:
        data[exp] = read_data(exp_dirs[exp])
        if exp not in used_exps:
            used_exps.append(exp)
    for valname in plot["vals"]:
        idx += 1
        ax = plt.subplot(2, 2, idx)

        ax.grid()
        method_index = 0
        y_min, y_max = 0, 0

        if valname == "reward":
            # find longest experiment
            _mx = 0
            for _exp in plot["exps"]:
                if data[_exp]["reward"].shape[0] > _mx:
                    _mx = data[_exp]["reward"].shape[0]
                    exp = _exp

            _y_min, _y_max = plot_line(ax, data[exp]["#sample"], data[exp]["reference reward"], window_size, alpha, colors["ref"], linewidth, lines["ref"])
            if "ref" not in used_exps:
                used_exps.append("ref")
            
            y_min = min(y_min, _y_min)
            y_max = max(y_max, _y_max)
        
        for exp in plot["exps"]:
            _y_min, _y_max = plot_line(ax, data[exp]["#sample"], data[exp][valname], window_size, alpha, colors[exp], linewidth, lines[exp])
            
            y_min = min(y_min, _y_min)
            y_max = max(y_max, _y_max)
        
        y_range = y_max - y_min
        ax.tick_params("x", labelsize=font_size)
        ax.tick_params("y", labelsize=font_size)
        ax.set_xlabel("#sample", size=font_size)
        ax.set_ylabel(valname, size=font_size)
        ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
        ax.set_title(str(idx), size=legend_font_size)

legend_elements = [Line2D([0], [0], lw=linewidth, label=exp, color=colors[exp], linestyle = lines[exp]) for exp in used_exps]
figure.legend(handles=legend_elements, loc='upper center', prop={'size': legend_font_size}, ncol=len(used_exps), bbox_to_anchor=anchor, frameon=False)

figure.tight_layout()
figure.savefig("plot.pdf", bbox_inches="tight", dpi=300)
plt.close(figure)