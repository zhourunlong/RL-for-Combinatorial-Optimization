from agent import *
from env import *
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
    print("Experiment dir : {}".format(logdir))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(logdir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    shutil.copy("agent.py", os.path.join(logdir, "code"))
    shutil.copy("env.py", os.path.join(logdir, "code"))
    shutil.copy("train.py", os.path.join(logdir, "code"))

    env = CSPEnv(args.n, args.batch_size)
    agent = NaiveAgent(args.n, args.lr)

    #running_reward, running_loss = 0, 0

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
            #running_reward += reward
            #running_loss += loss
        
            pbar.set_description("Episode: %8d, Reward: %2.4f, Loss: %2.4f" % (episode, reward, loss))

            #logging.info("Episode: %d, Reward: %0.4f, Loss: %0.4f" % (episode, np.mean(running_reward), np.mean(running_loss)))

            if (episode + 1) % args.save_episode == 0:
                savepath = os.path.join(logdir, "models/%d.pt" % (episode))
                torch.save(agent, savepath)

    #env.print_v()