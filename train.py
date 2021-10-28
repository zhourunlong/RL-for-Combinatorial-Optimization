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
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--n", default=10, type=int)
    parser.add_argument("--N", default=100, type=int)
    parser.add_argument("--d", default=10, type=int)
    parser.add_argument("--W", default=10, type=float)
    parser.add_argument("--save-episode", default=100, type=int)
    parser.add_argument("--phase-episode", default=2000, type=int)
    parser.add_argument("--seed", default=2018011309, type=int)
    parser.add_argument("--d0", default=10, type=int)
    parser.add_argument("--L", default=0, type=float)
    parser.add_argument("--curve-buffer-size", default=10, type=int)
    parser.add_argument("--type", default="uniform", choices=["uniform", "random"])
    parser.add_argument("--load-path", default=None, type=str)
    parser.add_argument("--rwd-succ", default=1, type=float)
    parser.add_argument("--rwd-fail", default=-1, type=float)
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

def collect_data(env, sampler, agent):
    agent.zero_grad()
    env.new_instance()

    csiz = env.bs // env.n
    rewards = torch.zeros((env.bs,), dtype=torch.double, device=env.device)
    grads_logp = torch.zeros((env.bs, agent.d), dtype=torch.double, device=env.device)
    idx = torch.zeros((env.bs,), dtype=torch.double, device=env.device)

    for i in range(1, env.n):
        il, ir = -(i + 1) * csiz, -i * csiz

        state = env.get_state()
        s_sampler, s_agent = state[:ir], state[il:]

        a_sampler, _ = sampler.get_action(s_sampler)
        a_agent, entropy = agent.get_action(s_agent)

        id = torch.randint(2, (csiz,), dtype=torch.bool, device=env.device)
        idx[il:ir] = id
        ax = torch.where(id, a_sampler[-csiz:], a_agent[:csiz])
        
        action = torch.cat((a_sampler[:-csiz], ax, a_agent[csiz:]))
        reward, active = env.get_reward(action)

        log_prob, grad_logp = agent.query_sa(s_sampler[-csiz:], a_sampler[-csiz:])
        rewards[il:ir] = active[il:ir] * torch.where(id, reward[il:ir] - agent.L * log_prob, reward[il:ir] + agent.L * entropy[:csiz])
        grads_logp[il:ir] = grad_logp * active[il:ir].unsqueeze(-1)
        rewards[ir:] += reward[ir:] + agent.L * active[ir:] * entropy[csiz:]
    
    rewards *= 4 * idx - 2
    
    #for i in range(env.n - 1, -1, -1):
    #    il, ir = i * csiz, (i + 1) * csiz
    #    agent.store_grad(rewards[il:ir], grads_logp[il:ir])
    agent.store_grad(rewards, grads_logp)

def evaluate(env, agent):
    env.new_instance()
    rewards = torch.zeros((env.bs,), dtype=torch.double, device=env.device)

    for i in range(env.n):
        state = env.get_state()
        action, _ = agent.get_action(state)
        reward, _ = env.get_reward(action)
        
        rewards += reward
    
    return rewards.mean().cpu().numpy()

if __name__ == "__main__":
    args = get_args()

    #args.lr = 2 / sqrt(2 * args.d0 * args.phase_episode)

    assert (args.N - args.n) % args.d == 0
    assert args.save_episode % args.curve_buffer_size == 0

    set_seed(args.seed)

    logdir = "Experiment-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "code"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "result"), exist_ok=True)
    print("Experiment dir: {}".format(logdir))

    dirs = os.listdir(".")
    for fn in dirs:
        if os.path.splitext(fn)[-1] == ".py":
            shutil.copy(fn, os.path.join(logdir, "code"))

    if args.load_path is not None:
        package = torch.load(args.load_path, map_location=args.device)
        envs = package["envs"]
        agent = package["agent"]
        for e in envs:
            e.move_device(args.device)
        agent.move_device(args.device)
        
        env = envs[0]
        phi = agent.get_phi_all()
        idx = opt_tabular(env.probs.cpu().numpy())
        policy_star = torch.zeros((env.n, 2), dtype=torch.double, device=args.device)
        for i in idx:
            policy_star[i - 1, 1] = 1
        kappa = calc_kappa(env.probs, policy_star, agent.get_policy(), phi)

        #print(kappa)

        #print(agent.theta, torch.norm(agent.theta))

        f = torch.arange(1, agent.n + 1, dtype=torch.double, device=args.device) / agent.n
        states_1 = torch.stack((f, torch.ones_like(f)), dim=1)
        states_0 = torch.stack((f, torch.zeros_like(f)), dim=1)
        logits_1 = agent.get_logits(states_1).view(-1,).cpu()
        logits_0 = agent.get_logits(states_0).view(-1,).cpu()

        #print(logits_1, logits_0)

        plot_prob_fig(agent, env, "visualize.jpg", args.device)

        args.n = env.n
    else:
        env = CSPEnv(args.type, args.device, args.rwd_succ, args.rwd_fail)
        agent = CSPLogLinearAgent(args.lr, args.d0, args.L, args.W, args.device)
        
        envs = []
        for n in range(args.N, args.n - args.d, -args.d):
            env.set_n(n)
            envs.append(copy.deepcopy(env))
        envs.reverse()

    envs.insert(0, None)

    n = args.n - args.d
    num_episode = ((args.N - args.n) // args.d + 1) * args.phase_episode

    running_reward = [0 for _ in range(num_episode // args.curve_buffer_size)]
    running_kappa = [0 for _ in range(num_episode // args.curve_buffer_size)]
    reward_buf, kappa_buf = 0, 0

    with tqdm(range(num_episode), desc="Training") as pbar:
        for episode in pbar:
            if episode % args.phase_episode == 0:
                n += args.d
                agent.update_n(n)
                envs.pop(0)
                env = envs[0]
                env.set_bs(n * args.batch_size)

                if episode > 0:
                    sampler = copy.deepcopy(agent)
                else:
                    sampler = agent
                #sampler = agent

                phi = agent.get_phi_all()
                idx = opt_tabular(env.probs.cpu().numpy())
                policy_star = torch.zeros((n, 2), dtype=torch.double, device=args.device)
                for i in idx:
                    policy_star[i - 1, 1] = 1
            
            collect_data(env, sampler, agent)
            agent.update_param()

            reward = evaluate(env, agent)
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
                package = {"agent":agent, "envs":envs}
                torch.save(package, os.path.join(logdir, "checkpoint/%08d.pt" % (episode)))

                plot_prob_fig(agent, env, os.path.join(logdir, "result/visualize%08d.jpg" % (episode)), args.device)
                
                len = (episode + 1) // args.curve_buffer_size
                plot_rl_fig(running_reward[:len], "Reward", np.log(running_kappa[:len]), "log(Kappa)", os.path.join(logdir, "result/curve.jpg"), args.curve_buffer_size)
