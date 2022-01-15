import pickle
from scipy.stats import sem ,t
from scipy import mean
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import json


color = {
    'cgmix_5': [216, 30, 54],  # Red
#    'cgmix.3': [160, 32, 240],
    'cgmix1': [254, 151, 0],
    'dcg': [0, 178, 238],
    'cgmix_3_lr': [160, 32, 240],
    'cgmix_5_b': [180, 180, 180],  #
    'cgmix.3': [139, 101, 8],
    'cgmix_2_lr': [0, 100, 0],
    'qmix': [204, 153, 255],
}

color = {key: np.array(value, np.float) / 255. for key, value in color.items()}

algorithms = [
    'cgmix1',
    'cgmix_5',
    'cgmix_3_lr',
#    'cgmix_5_b',
    'dcg',
    # 'qplex',
]

algorithm_labels = [
    'cgmix_1.0',
    'cgmix_0.5',
    'cgmix_3_lr',
#    'cgmix_5_b',
    'DCG',
    # 'QPLEX',
]

algorithm_lines = [
    '-',
    '-',
    '-',
    '-',
    '--',
    '--',
    '--',
]

envs = [
        'pursuit',
]

env_labels = [
              'pursuit',
]

env_data = {
    'pursuit' : 'test_prey_left_mean',
}

env_length = {
    'pursuit' : 201,
}

# #################

alpha = 0.2
scale = 50.
confidence = 0.95
log_scale = False
font_size = 26
legend_font_size = 32
anchor = (0.5, 1.08)


def smooth(data, data_name):
    if data_name == 'test_return_mean':
        start = 0.0
    elif data_name == 'test_trans_mean':
        start = 5.0
    else:
        start = 3.0
    range1 = 10.0
    new_data = np.zeros_like(data)
    for i in range(int(start), int(range1)):
        new_data[i] = 1. * sum(data[0 : i + 1]) / (i + 1)
    for i in range(int(range1), len(data)):
        new_data[i] = 1. * sum(data[i - int(range1) + 1 : i + 1]) / range1

    return new_data


# def resize(data):
#     if len(data) < max_length:
#         data += [0 for _ in range(max_length - len(data))]
#     elif len(data) > max_length:
#         data = data[:max_length]
#
#     return data


def read_data(env, alg, data_name, cut, cut_length=None):
    data_n = []
    x_n = []

    files = []
    path = alg
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith('info.json'):
                files.append(os.path.join(r, file))

    run_number = len(files)

    for f in files:
        with open(f, 'r') as _f:
            print(f)
            d = json.load(_f)

            if isinstance(d[data_name][0], float):
                data_n.append(np.array(d[data_name]))
            else:
                new_d = []
                for data in d[data_name]:
                    new_d.append(data['value'])
                data_n.append(np.array(new_d))
            x_n.append(np.array(d[data_name+'_T']))

    print([len(_x) for _x in x_n])

    min_length = min([len(_x) for _x in x_n] + [cut_length])
    
    data_n = [data[:min_length] for data in data_n]
    x_n = [x[:min_length] for x in x_n]

    data_n = np.array(data_n)
    if data_name == 'test_battle_won_mean':
        data_n = data_n * 100
    if data_name == 'test_prey_left_mean':
        data_n = 5 - data_n
    if data_name == 'test_return_mean':
        data_n = 10 + data_n

    return np.array(x_n), data_n, min_length, run_number


s_cut = 2000

if __name__ == '__main__':
    # figure =
    # ######### 10
    figure = None
    figure = plt.figure(figsize=(32, 11))

    data = [[] for _ in algorithms]
    
    print(algorithms)

    legend_elements = [Line2D([0], [0], lw=4, label=label, color=color[alg], linestyle = style) for (alg, label, style) in
                       zip(algorithms, algorithm_labels, algorithm_lines)]
    figure.legend(handles=legend_elements, loc='upper center', prop={'size': legend_font_size}, ncol=min(len(algorithms), 3),
                  bbox_to_anchor=(0.5, 1.15), frameon=False)

    for idx, env in enumerate(envs):
        ax = None
        ax= plt.subplot(1, 1, idx+1)

        ax.grid()
        # figure = plt.figure()
        # plt.grid()
        method_index = 0
        
        print(zip(algorithms, algorithm_labels, algorithm_lines))
        print(algorithms, algorithm_labels,algorithm_lines)

        for (alg, label, style) in zip(algorithms, algorithm_labels, algorithm_lines):
            print(alg)
            x, y, min_length, run_number = read_data(env, alg, env_data[env], cut='fix_cut', cut_length=env_length[env])
            print(env, alg, run_number)

            if run_number == 0:
                continue

            if env == 'pursuit':
                print(alg, np.median(y[:, -11:].max(axis=-1), axis=0))

            y_mean = smooth(np.median(y, axis=0), env_data[env])#y_mean = smooth(np.mean(y, axis=0))
            train_scores_mean = y_mean
            data[method_index].append(y_mean[:s_cut])
            method_index += 1

            low = smooth(np.percentile(y, 25, axis=0), env_data[env])
            high = smooth(np.percentile(y, 75, axis=0), env_data[env])

            # h = smooth(sem(y) * t.ppf((1 + confidence) / 2, min_length - 1))
            #     h = smooth(sem(data) * t.ppf((1 + confidence) / 2, max_length - 1))
            # bhos = np.linspace(1, min_length, min_length)
            bhos = x[0] / 1000000
            # if log_scale:
            #     train_scores_mean = np.log(train_scores_mean + scale) - np.log(scale)
            #     h = np.log(h + scale) - np.log(scale)
            ax.fill_between(bhos, low,
                             high, alpha=alpha,
                             color=color[alg], linewidth=0)
            width = 4
            if alg == 'dsco':
                width = 4.5
            ax.plot(bhos, train_scores_mean, color=color[alg], label=label, linewidth=width, linestyle=style)

        # Others
        ax.tick_params('x', labelsize=font_size)
        ax.tick_params('y', labelsize=font_size)
        ax.set_xlabel('T (mil)', size=font_size)
        ax.set_title(env_labels[idx], size=legend_font_size)
        if env_data[env] == 'test_battle_won_mean':
            ax.set_ylabel('Test Win %', size=font_size)
            ax.set_ylim(-5, 105)
        elif env_data[env] == 'test_trans_mean':
            ax.set_ylabel('Test Transmitted', size=font_size)
            ax.set_ylim(-5, 70)
        elif env_data[env] == 'test_scaned_mean':
            ax.set_ylabel('Test Captured', size=font_size)
            ax.set_ylim(-2, 32)
        elif env_data[env] == 'test_prey_left_mean':
            ax.set_ylabel('Test Prey Caught', size=font_size)
            ax.set_ylim(-0.1, 5.1)
        elif env_data[env] == 'test_return_mean':
            ax.set_ylabel('Test Match Score', size=font_size)
            ax.set_ylim(5.8, 10.2)

    # figure.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=anchor,
    #               prop={'size': legend_font_size}, ncol=min(len(methods), 4), frameon=False)

    figure.tight_layout()
    # plt.show()

    # plt.gca().set_facecolor([248./255, 248./255, 255./255])
    figure.savefig('./maco.pdf', bbox_inches='tight', dpi=300)  # , bbox_extra_artists=(lgd,)
    plt.close(figure)
