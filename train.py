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
    parser.add_argument("--batch-size", default=10000, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num-episode", default=20000, type=int)
    parser.add_argument("--n", default=10, type=int)
    parser.add_argument("--save-episode", default=1000, type=int)
    parser.add_argument("--phase-episode", default=2000, type=int)
    parser.add_argument("--seed", default=2018011309, type=int)
    parser.add_argument("--regular-lambda", default=0, type=float)
    parser.add_argument("--loglinear-d0", default=20, type=int)
    parser.add_argument("--curve-buffer-size", default=100, type=int)
    parser.add_argument("--type", default="uniform", choices=["uniform", "random"])
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

    assert args.save_episode % args.curve_buffer_size == 0

    set_seed(args.seed)

    logdir = "Experiment-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "code"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "results"), exist_ok=True)
    print("Experiment dir: {}".format(logdir))

    dirs = os.listdir(".")
    for fn in dirs:
        if os.path.splitext(fn)[-1] == ".py":
            shutil.copy(fn, os.path.join(logdir, "code"))
    
    n = args.n

    env = CSPEnv(n, args.batch_size, args.type)
    agent = LogLinearAgent(n, args.lr, args.regular_lambda, args.loglinear_d0)
    #agent = NeuralNetworkAgent(n, args.lr, args.regular_lambda)

    running_reward, running_loss = [0 for _ in range(args.num_episode // args.curve_buffer_size)], [0 for _ in range(args.num_episode // args.curve_buffer_size)]
    reward_buf, loss_buf = 0, 0

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
            
            #print(env.probs)
            #print(env.v)
            #print("argmax", env.argmax)

            states, rewards, probs, log_probs, entropies, grads_logp = [], [], [], [], [], []
            for step in range(n):
                state = env.get_state()
                action, prob, log_prob, entropy, grad_logp = agent.get_action(state)
                reward = env.get_reward(action)
                
                states.append(state)
                rewards.append(reward)
                probs.append(prob)
                log_probs.append(log_prob)
                entropies.append(entropy)
                grads_logp.append(grad_logp)
            
            reward, loss = agent.update_param(states, rewards, probs, log_probs, entropies, grads_logp)
            reward_buf += reward
            loss_buf += loss
            if (episode + 1) % args.curve_buffer_size == 0:
                idx = episode // args.curve_buffer_size
                running_reward[idx] = reward_buf / args.curve_buffer_size
                running_loss[idx] = loss_buf / args.curve_buffer_size
                reward_buf, loss_buf = 0, 0
        
            pbar.set_description("Epi: %d, N: %d, R: %2.4f, L: %2.4f" % (episode, n, reward, loss))

            if (episode + 1) % args.save_episode == 0:
                savepath = os.path.join(logdir, "models/%08d.pt" % (episode))
                torch.save(agent, savepath)
                plot_prob_fig(agent, env, os.path.join(logdir, "results/visualize%08d.jpg" % (episode)))
                plot_rl_fig(running_reward, running_loss, os.path.join(logdir, "results/curve.jpg"), args.curve_buffer_size, (episode + 1) // args.curve_buffer_size)
            
            current_n_episode += 1

    #env.print_v()