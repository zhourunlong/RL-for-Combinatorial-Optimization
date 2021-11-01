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
    parser.add_argument("--batch-size", default=10000, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--n", default=10, type=int)
    parser.add_argument("--N", default=100, type=int)
    parser.add_argument("--d", default=90, type=int)
    parser.add_argument("--W", default=10, type=float)
    parser.add_argument("--save-episode", default=1000, type=int)
    parser.add_argument("--phase-episode", default=10000, type=int)
    parser.add_argument("--seed", default=2018011309, type=int)
    parser.add_argument("--d0", default=10, type=int)
    parser.add_argument("--L", default=0, type=float)
    parser.add_argument("--curve-buffer-size", default=100, type=int)
    parser.add_argument("--distr-type", default="uniform", choices=["uniform", "random"])
    parser.add_argument("--sample-type", default="pi^t", choices=["pi^0", "pi^t", "curriculum"])
    parser.add_argument("--load-path", default=None, type=str)
    parser.add_argument("--rwd-succ", default=1, type=float)
    parser.add_argument("--rwd-fail", default=0, type=float)
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

def unpack(agent, envs, best, agent_best, sampler):
    return agent, envs, best, agent_best, sampler

def collect_data(env, sampler, agent):
    agent.zero_grad()
    env.new_instance()

    csiz = env.bs // (env.n + 1)
    rewards = torch.zeros((env.bs,), dtype=torch.double, device=env.device)
    log_probs, idx, actives = torch.zeros_like(rewards), torch.zeros_like(rewards), torch.zeros_like(rewards)
    grads_logp = torch.zeros((env.bs, agent.d), dtype=torch.double, device=env.device)

    for i in range(env.n):
        il, ir = -(i + 2) * csiz, -(i + 1) * csiz

        state = env.get_state()
        s_sampler, s_agent = state[:ir], state[il:]

        a_sampler, _ = sampler.get_action(s_sampler)
        a_agent, entropy = agent.get_action(s_agent)

        id = torch.randint(2, (csiz,), dtype=torch.double, device=env.device)
        idx[il:ir] = id
        ax = id * a_sampler[-csiz:] + (1 - id) * a_agent[:csiz]
        
        action = torch.cat((a_sampler[:-csiz], ax, a_agent[csiz:]))
        reward, active = env.get_reward(action)

        actives[il:ir] = active[il:ir]
        log_probs[il:ir], grad_logp = agent.query_sa(s_sampler[-csiz:], a_sampler[-csiz:])
        grads_logp[il:ir] = grad_logp * active[il:ir].unsqueeze(-1)
        
        rewards[il:ir] = active[il:ir] * (reward[il:ir] + agent.L * (1 - id) * entropy[:csiz])
        rewards[ir:-csiz] += reward[ir:-csiz] + agent.L * active[ir:-csiz] * entropy[csiz:-csiz]
        rewards[-csiz:] += reward[-csiz:]
    
    idx[-csiz:] = 0.75
    rewards = rewards * (4 * idx - 2) - agent.L * actives * log_probs
    agent.store_grad(rewards[:-csiz], grads_logp[:-csiz])

    return rewards[-csiz:].mean().detach().cpu()

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
        agent, envs, best, agent_best, sampler = unpack(**package)

        for it in envs + [agent, agent_best, sampler]:
            it.move_device(args.device)
        
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
        env = CSPEnv(args.distr_type, args.device, args.rwd_succ, args.rwd_fail)
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
                env.set_bs((n + 1) * args.batch_size) # one more batch for evaluation

                if args.load_path is None:
                    best = -1e9
                    if args.sample_type == "pi^0":
                        sampler = copy.deepcopy(agent)
                    elif args.sample_type == "pi^t":
                        sampler = agent
                    else:
                        if episode > 0:
                            sampler = copy.deepcopy(agent)
                        else:
                            sampler = agent
                elif episode > 0:
                    best = -1e9
                    if args.sample_type == "pi^t":
                        sampler = agent
                    else:
                        sampler = copy.deepcopy(agent)

                phi = agent.get_phi_all()
                idx = opt_tabular(env.probs.cpu().numpy())
                policy_star = torch.zeros((n, 2), dtype=torch.double, device=args.device)
                for i in idx:
                    policy_star[i - 1, 1] = 1
            
            reward = collect_data(env, sampler, agent)
            agent.update_param()
            reward_buf += reward
            if reward > best:
                best = reward
                agent_best = copy.deepcopy(agent)
                best_changed = True

            kappa = calc_kappa(env.probs, policy_star, agent.get_policy(), phi).cpu().numpy()
            kappa_buf += kappa

            if (episode + 1) % args.curve_buffer_size == 0:
                idx = episode // args.curve_buffer_size
                running_reward[idx] = reward_buf / args.curve_buffer_size
                running_kappa[idx] = kappa_buf / args.curve_buffer_size
                reward_buf, kappa_buf = 0, 0
        
            pbar.set_description("Epi: %d, N: %d, R: %2.4f, K: %3.3f" % (episode, n, reward, kappa))

            if (episode + 1) % args.save_episode == 0:
                del env.v
                package = {"agent":agent, "envs":envs, "best": best, "agent_best":agent_best, "sampler":sampler}
                torch.save(package, os.path.join(logdir, "checkpoint/%08d.pt" % (episode)))

                plot_prob_fig(agent, env, os.path.join(logdir, "result/visualize%08d.jpg" % (episode)), args.device)
                if best_changed:
                    plot_prob_fig(agent_best, env, os.path.join(logdir, "result/visualize_best.jpg"), args.device)
                    best_changed = False
                
                len = (episode + 1) // args.curve_buffer_size
                plot_rl_fig(running_reward[:len], "Reward", np.log(running_kappa[:len]), "log(Kappa)", os.path.join(logdir, "result/curve.jpg"), args.curve_buffer_size)
