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
from calculate_opt_policy import *
from calculate_kappa import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=10000, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
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

if __name__ == "__main__":
    args = get_args()

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

        args.n = env.n
    else:
        env = CSPEnv(args.batch_size, args.type)
        agent = LogLinearAgent(args.lr, args.loglinear_d0, args.W)

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

                phi = agent.get_phi_all()
                idx = opt_tabular(env.probs.cpu().numpy())
                policy_star = torch.zeros((n, 2), dtype=torch.double, device="cuda")
                for i in idx:
                    policy_star[i - 1, 1] = 1
            else:
                env.reset(False)

            acts, rewards, reward0s, phis = [], [], [], []
            for step in range(n):
                state = env.get_state()
                action, phit = agent.get_action(state)
                print(action)
                active, reward, reward0 = env.get_reward(action)
                
                acts.append(active)
                rewards.append(reward)
                reward0s.append(reward0)
                phis.append(phit)
            
            reward = agent.update_param(acts, rewards, reward0s, phis)
            reward_buf += reward

            kappa = calc_kappa(env.probs, policy_star, agent.get_policy(), phi).detach().cpu()
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
                plot_rl_fig(running_reward, running_kappa, os.path.join(logdir, "results/curve.jpg"), args.curve_buffer_size, (episode + 1) // args.curve_buffer_size)
