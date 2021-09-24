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
import time
import copy
from math import sqrt
from calculate_opt_policy import *
from calculate_kappa import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=5000, type=int)
    #parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--n", default=10, type=int)
    parser.add_argument("--N", default=100, type=int)
    parser.add_argument("--d", default=10, type=int)
    parser.add_argument("--W", default=1000, type=float)
    parser.add_argument("--save-episode", default=1000, type=int)
    parser.add_argument("--phase-episode", default=10000, type=int)
    parser.add_argument("--seed", default=2018011309, type=int)
    parser.add_argument("--loglinear-d0", default=20, type=int)
    parser.add_argument("--curve-buffer-size", default=100, type=int)
    parser.add_argument("--type", default="uniform", choices=["uniform", "random"])
    parser.add_argument("--load-path", default=None, type=str)
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

def collect_data(agent, env, rewards_only=False):
    state0s, state1s, actions, rewards, rs0s, acts, probs = [], [], [], [], [], [], []

    for step in range(env.n):
        state0, state1 = env.get_state()
        action, prob = agent.get_action((state0, state1))
        reward, rs0, active = env.get_reward(action)
        
        rewards.append(reward)
        if rewards_only == False:
            state0s.append(state0)
            state1s.append(state1)
            actions.append(action)
            rs0s.append(rs0)
            acts.append(active)
            probs.append(prob)
        #log_probs.append(log_prob)
        #grads_logp.append(grad_logp)
    
    if rewards_only:
        return rewards
    else:
        return (state0s, state1s), actions, rewards, rs0s, acts, probs

if __name__ == "__main__":
    args = get_args()

    lr = 2 / sqrt(2 * args.loglinear_d0 * args.phase_episode)

    assert (args.N - args.n) % args.d == 0
    assert args.save_episode % args.curve_buffer_size == 0

    if args.load_path is not None:
        package = torch.load(args.load_path)
        env = package["env"]
        agent = package["agent"]
        phi = agent.get_phi_all()
        idx = opt_tabular(env.probs.cpu().numpy())
        policy_star = torch.zeros((env.n, 2), dtype=torch.double, device="cuda")
        for i in idx:
            policy_star[i - 1, 1] = 1
        kappa = calc_kappa(env.probs, policy_star, agent.get_policy(), phi)

        print(kappa)

        print(agent.theta, torch.norm(agent.theta))

        states_1 = [torch.arange(1, agent.n + 1, dtype=torch.double, device="cuda") / agent.n, torch.ones((agent.n,), dtype=torch.double, device="cuda")]
        states_0 = [torch.arange(1, agent.n + 1, dtype=torch.double, device="cuda") / agent.n, torch.zeros((agent.n,), dtype=torch.double, device="cuda")]
        logits_1 = agent.get_logits(states_1).view(-1,).cpu()
        logits_0 = agent.get_logits(states_0).view(-1,).cpu()

        print(logits_1, logits_0)

        plot_prob_fig(agent, env, "visualize.jpg")

        #(success, x) = opt_loglinear(env.n, agent.d0, 5, (idx[-1] - 0.5) / env.n)
        #x = torch.tensor(x).double().cuda()
        #agent.theta = x.view_as(agent.theta)

        args.n = env.n
    else:
        env = CSPEnv(args.batch_size, args.type)
        agent = LogLinearAgent(lr, args.loglinear_d0, args.W)

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
    
    n = args.n - args.d
    num_episode = ((args.N - args.n) // args.d + 1) * args.phase_episode

    running_reward = [0 for _ in range(num_episode // args.curve_buffer_size)]
    running_kappa = [0 for _ in range(num_episode // args.curve_buffer_size)]
    reward_buf, kappa_buf = 0, 0

    with tqdm(range(num_episode), desc="Training") as pbar:
        for episode in pbar:
            if episode % args.phase_episode == 0:
                n += args.d
                env.reset(True, n)
                agent.update_n(n)
                if episode > 0:
                    agent0 = copy.deepcopy(agent)
                else:
                    agent0 = agent

                phi = agent.get_phi_all()
                idx = opt_tabular(env.probs.cpu().numpy())
                policy_star = torch.zeros((n, 2), dtype=torch.double, device="cuda")
                for i in idx:
                    policy_star[i - 1, 1] = 1
            else:
                env.reset(False)

            #print(env.probs)
            states, actions, rewards, rs0s, acts, probs = collect_data(agent0, env)
            agent.update_param(states, actions, rs0s, acts, probs)

            env.reset_i()
            rewards = collect_data(agent, env, True)
            reward = (torch.stack(rewards).sum() / args.batch_size).cpu().numpy()
            reward_buf += reward

            kappa = calc_kappa(env.probs, policy_star, agent.get_policy(), phi).cpu().numpy()
            kappa_buf += kappa

            if (episode + 1) % args.curve_buffer_size == 0:
                idx = episode // args.curve_buffer_size
                running_reward[idx] = reward_buf / args.curve_buffer_size
                running_kappa[idx] = kappa_buf / args.curve_buffer_size
                reward_buf, kappa_buf = 0, 0
        
            pbar.set_description("Epi: %d, N: %d, R: %2.4f, K: %3.3f" % (episode, n, reward, kappa))

            if (episode + 1) % args.save_episode == 0:
                savepath = os.path.join(logdir, "models/%08d.pt" % (episode))
                package = {"agent":agent, "env":env}
                torch.save(package, savepath)
                plot_prob_fig(agent, env, os.path.join(logdir, "results/visualize%08d.jpg" % (episode)))
                plot_rl_fig(running_reward, "Reward", running_kappa, "Kappa", os.path.join(logdir, "results/curve.jpg"), args.curve_buffer_size, (episode + 1) // args.curve_buffer_size)
