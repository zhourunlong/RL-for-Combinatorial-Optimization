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
import configparser

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--config", default="config.ini", type=str)
    parser.add_argument("--problem", choices=["CSP", "OLKnapsack"], required=True)
    parser.add_argument("--save-episode", default=1000, type=int)
    parser.add_argument("--phase-episode", default=10000, type=int)
    parser.add_argument("--seed", default=2018011309, type=int)
    parser.add_argument("--curve-buffer-size", default=100, type=int)
    parser.add_argument("--sample-type", default="pi^t", choices=["pi^0", "pi^t", "curriculum"])
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

def unpack_config(batch_size, n_start, n_end, step, **kwargs):
    return int(batch_size), int(n_start), int(n_end), int(step)

def unpack_checkpoint(agent, envs, best, agent_best, sampler, **kwargs):
    return agent, envs, best.cpu(), agent_best, sampler

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

    assert args.save_episode % args.curve_buffer_size == 0

    if args.load_path is not None:
        args.config = os.path.join(os.path.dirname(args.load_path), "../config.ini")

    parser = configparser.RawConfigParser()
    parser.optionxform = lambda option: option
    parser.read(args.config, encoding='utf-8')
    config = dict(parser.items(args.problem))
    batch_size, n_start, n_end, step = unpack_config(**config)

    set_seed(args.seed)

    assert (n_end - n_start) % step == 0

    logdir = "Exp-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.problem)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "code"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "result"), exist_ok=True)
    print("Experiment dir: {}".format(logdir))

    dirs = os.listdir(".")
    for fn in dirs:
        if os.path.splitext(fn)[-1] == ".py":
            shutil.copy(fn, os.path.join(logdir, "code"))
    shutil.copy(args.config, os.path.join(logdir, "config.ini"))

    if args.load_path is not None:
        package = torch.load(args.load_path, map_location=args.device)
        agent, envs, best, agent_best, sampler = unpack_checkpoint(**package)

        for it in envs + [agent, agent_best, sampler]:
            it.move_device(args.device)
        
        #env = envs[0]
        #phi = agent.get_phi_all()
        #idx = opt_tabular(env.probs.cpu().numpy())
        #policy_star = torch.zeros((env.n, 2), dtype=torch.double, device=args.device)
        #for i in idx:
        #    policy_star[i - 1, 1] = 1

        #kappa = calc_kappa(env.probs, policy_star, agent.get_policy(), phi)

        #print(kappa)

        #print(agent.theta, torch.norm(agent.theta))

        #f = torch.arange(1, agent.n + 1, dtype=torch.double, device=args.device) / agent.n
        #states_1 = torch.stack((f, torch.ones_like(f)), dim=1)
        #states_0 = torch.stack((f, torch.zeros_like(f)), dim=1)
        #logits_1 = agent.get_logits(states_1).view(-1,).cpu()
        #logits_0 = agent.get_logits(states_0).view(-1,).cpu()

        #print(logits_1, logits_0)

        #plot_prob_fig(agent, env, "visualize.jpg", args.device)

        n_start = envs[0].n
    else:
        if args.problem == "CSP":
            env = CSPEnv(args.device, **config)
            agent = CSPAgent(args.device, **config)
        elif args.problem == "OLKnapsack":
            env = OLKnapsackEnv(args.device, **config)
            agent = OLKnapsackAgent(args.device, **config)
        
        envs = []
        for n in range(n_end, n_start - step, -step):
            env.set_n(n)
            envs.append(copy.deepcopy(env))
        envs.reverse()

    envs.insert(0, None)

    n = n_start - step
    num_episode = ((n_end - n_start) // step + 1) * args.phase_episode

    arr_size = num_episode // args.curve_buffer_size
    num_samples = np.zeros(arr_size, dtype=np.uint64)
    running_reward = np.zeros(arr_size)
    running_kappa = np.zeros(arr_size)
    cnt_samples, reward_buf, kappa_buf = 0, 0, 0

    with tqdm(range(num_episode), desc="Training") as pbar:
        for episode in pbar:
            if episode % args.phase_episode == 0:
                n += step
                agent.update_n(n)
                envs.pop(0)
                env = envs[0]
                env.set_bs((n + 1) * batch_size) # one more batch for evaluation

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

                #pi_star = env.get_opt_policy()
                #phi = agent.get_phi_all()
            
            reward = collect_data(env, sampler, agent)
            cnt_samples += n * batch_size
            agent.update_param()
            reward_buf += reward
            if reward > best:
                best = reward
                agent_best = copy.deepcopy(agent)
                best_changed = True

            #kappa = calc_kappa(env.probs, pi_star, agent.get_policy(), phi).cpu().numpy()
            kappa = 1
            kappa_buf += kappa

            if (episode + 1) % args.curve_buffer_size == 0:
                idx = episode // args.curve_buffer_size
                num_samples[idx] = cnt_samples
                running_reward[idx] = reward_buf / args.curve_buffer_size
                running_kappa[idx] = kappa_buf / args.curve_buffer_size
                reward_buf, kappa_buf = 0, 0
        
            pbar.set_description("Epi: %d, N: %d, R: %2.4f, K: %3.3f" % (episode, n, reward, kappa))

            if (episode + 1) % args.save_episode == 0:
                env.clean()
                package = {"agent":agent, "envs":envs, "best": best, "agent_best":agent_best, "sampler":sampler}
                torch.save(package, os.path.join(logdir, "checkpoint/%08d.pt" % (episode)))

                #plot_prob_fig(agent, env, os.path.join(logdir, "result/visualize%08d.jpg" % (episode)), args.device)
                #if best_changed:
                    #plot_prob_fig(agent_best, env, os.path.join(logdir, "result/visualize_best.jpg"), args.device)
                    #best_changed = False
                
                len = (episode + 1) // args.curve_buffer_size
                plot_rl_fig(np.arange(1, len + 1) * args.curve_buffer_size, "Episodes", running_reward[:len], "Reward", np.log(running_kappa[:len]), "log(Kappa)", os.path.join(logdir, "result/curve_episode.jpg"))
                plot_rl_fig(num_samples[:len], "Samples", running_reward[:len], "Reward", np.log(running_kappa[:len]), "log(Kappa)", os.path.join(logdir, "result/curve_sample.jpg"))
