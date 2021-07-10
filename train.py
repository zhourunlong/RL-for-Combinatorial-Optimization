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
    parser.add_argument("--seed", default=2018011309, type=int)
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

    '''
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(logdir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    '''

    shutil.copy("agent.py", os.path.join(logdir, "code"))
    shutil.copy("env.py", os.path.join(logdir, "code"))
    shutil.copy("train.py", os.path.join(logdir, "code"))
    shutil.copy("visualize.py", os.path.join(logdir, "code"))

    env = CSPEnv(args.n, args.batch_size)
    agent = NaiveAgent(args.n, args.lr)

    running_reward, running_loss = [], []

    with tqdm(range(args.num_episode), desc="Training") as pbar:
        for episode in pbar:
            env.reset(True)

            rewards, log_probs, entropies = [], [], []
            for step in range(args.n):
                state = env.get_state()
                action, log_prob, entropy = agent.get_action(state)
                reward = env.get_reward(action)
                
                rewards.append(reward)
                log_probs.append(log_prob)
                entropies.append(entropy)
            
            reward, loss = agent.update_param(rewards, log_probs, entropies)
            running_reward.append(reward)
            running_loss.append(loss)
        
            pbar.set_description("Epi: %8d, R: %2.4f, L: %2.4f" % (episode, reward, loss))

            #logging.info("Epi: %d, R: %0.4f, L: %0.4f" % (episode, np.mean(running_reward), np.mean(running_loss)))

            if (episode + 1) % args.save_episode == 0:
                savepath = os.path.join(logdir, "models/%08d.pt" % (episode))
                torch.save(agent, savepath)
                plot_prob_fig(agent, os.path.join(logdir, "results/visualize%08d.jpg" % (episode)))
                plot_rl_fig(running_reward, running_loss, os.path.join(logdir, "results/curve.jpg"))

    #env.print_v()