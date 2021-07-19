from agent import *
from env import *
from visualize import *
import argparse
import os, sys, logging, time
from tqdm import tqdm
import shutil
import numpy as np
import torch
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num-episode", default=10000, type=int)
    parser.add_argument("--n", default=100, type=int)
    parser.add_argument("--save-episode", default=1000, type=int)
    parser.add_argument("--phase-episode", default=1000, type=int)
    parser.add_argument("--seed", default=2018011309, type=int)
    parser.add_argument("--regular-lambda", default=0.0001, type=float)
    return parser.parse_args()

def set_seed(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)

    logdir = "Experiment-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "code"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "results"), exist_ok=True)
    print("Experiment dir: {}".format(logdir))

    shutil.copy("agent.py", os.path.join(logdir, "code"))
    shutil.copy("env.py", os.path.join(logdir, "code"))
    shutil.copy("train.py", os.path.join(logdir, "code"))
    shutil.copy("visualize.py", os.path.join(logdir, "code"))

    n = args.n

    env = CSPEnv(n, args.batch_size)
    agent = NeuralNetworkAgent(n, args.lr, args.regular_lambda)

    running_reward, running_loss = [], []

    current_n_episode = 0

    with tqdm(range(args.num_episode), desc="Training") as pbar:
        for episode in pbar:
            if current_n_episode >= args.phase_episode:
                n += 10
                current_n_episode = 0
                env.reset(True, n)
                agent.update_n(n)
            else:
                env.reset(True)

            states, rewards, probs, log_probs, entropies = [], [], [], [], []
            for step in range(n):
                state = env.get_state()
                action, prob, log_prob, entropy = agent.get_action(state)
                reward = env.get_reward(action)
                
                states.append(state)
                rewards.append(reward)
                probs.append(prob)
                log_probs.append(log_prob)
                entropies.append(entropy)
            
            reward, loss = agent.update_param(states, rewards, probs, log_probs, entropies)
            running_reward.append(reward)
            running_loss.append(loss)
        
            pbar.set_description("Epi: %d, N: %d, R: %2.4f, L: %2.4f" % (episode, n, reward, loss))

            if (episode + 1) % args.save_episode == 0:
                savepath = os.path.join(logdir, "models/%08d.pt" % (episode))
                torch.save(agent, savepath)
                plot_prob_fig(agent, os.path.join(logdir, "results/visualize%08d.jpg" % (episode)))
                plot_rl_fig(running_reward, running_loss, os.path.join(logdir, "results/curve.jpg"))
            
            current_n_episode += 1

    #env.print_v()