from agent import *
from env import *
from visualize import *
import argparse
import os, sys, logging, time
import shutil
import numpy as np
import torch
import random
import time
import copy
from math import log
from calculate_opt_policy import *
from calculate_kappa import *
import configparser
from logger import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--config", default="config.ini", type=str)
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

def unpack_config(sample_type, init_type, seed, grad_cummu_step, phase_episode, save_episode, smooth_episode, **kwargs):
    return sample_type, init_type, int(seed), int(grad_cummu_step), int(phase_episode), int(save_episode), int(smooth_episode)

def unpack_config_csp(n_start, n_end, **kwargs):
    return [[int(n_start)], [int(n_end)]]

def unpack_config_olkn(n_start, n_end, B_start, B_end, **kwargs):
    return [[int(n_start), float(B_start)], [int(n_end), float(B_end)]]

def unpack_checkpoint(agent, envs, sampler, **kwargs):
    return agent, envs, sampler

def collect_data(env, sampler, agent):
    env.new_instance()

    csiz = env.bs // (env.n + 1)
    rewards = torch.zeros((env.bs,), dtype=torch.double, device=env.device)
    log_probs, idx, actives = torch.zeros_like(rewards), torch.zeros_like(rewards), torch.zeros_like(rewards)
    grads_logp = torch.zeros((env.bs, agent.d), dtype=torch.double, device=env.device)

    for i in range(env.horizon):
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

    if args.load_path is not None:
        args.config = os.path.join(os.path.dirname(args.load_path), "../config.ini")

    parser = configparser.RawConfigParser()
    parser.optionxform = lambda option: option
    parser.read(args.config, encoding='utf-8')
    problem = dict(parser.items("Problem"))["name"]

    assert problem in ["CSP", "OLKnapsack"]

    config = dict(parser.items(problem))
    sample_type, init_type, seed, grad_cummu_step, phase_episode, save_episode, smooth_episode = unpack_config(**config)
    if problem == "CSP":
        curriculum_params = unpack_config_csp(**config)
    elif problem == "OLKnapsack":
        curriculum_params = unpack_config_olkn(**config)

    assert phase_episode % save_episode == 0
    assert save_episode % smooth_episode == 0
    assert sample_type in ["pi^0", "pi^t", "pi^t-pi^0"]
    assert init_type in ["pi^0", "pi^0-pi^t"]
    assert not (sample_type == "pi^t" and init_type == "pi^0"), "This will invalidate the first phase training!"

    set_seed(seed)

    logdir = "Exp-%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), problem)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "code"), exist_ok=True)
    for par in ["checkpoint", "logdata", "result"]:
        for sub in ["", "/warmup", "/final"]:
            os.makedirs(os.path.join(logdir, "%s%s" % (par, sub)), exist_ok=True)
    print("Experiment dir: %s" % (logdir))

    dirs = os.listdir(".")
    for fn in dirs:
        if os.path.splitext(fn)[-1] == ".py":
            shutil.copy(fn, os.path.join(logdir, "code"))
    shutil.copy(args.config, os.path.join(logdir, "config.ini"))
    
    logger = Logger(logdir)

    if args.load_path is not None:
        package = torch.load(args.load_path, map_location=args.device)
        agent, envs, sampler = unpack_checkpoint(**package)

        for it in envs + [agent, sampler]:
            it.move_device(args.device)
        
        env_curriculum_params = envs[0].curriculum_params
        
        if env_curriculum_params != curriculum_params[1]:
            curriculum_params = [env_curriculum_params, curriculum_params[1]]
        else:
            curriculum_params = [curriculum_params[1]]
    else:
        if problem == "CSP":
            env = CSPEnv(args.device, **config)
            agent = CSPAgent(args.device, **config)
        elif problem == "OLKnapsack":
            env = OLKnapsackEnv(args.device, **config)
            agent = OLKnapsackAgent(args.device, **config)
        
        envs = []

        if curriculum_params[0] == curriculum_params[1]:
            curriculum_params = [curriculum_params[0]]
        
        curriculum_params.reverse() # ensure that final environments (curriculum or not curriculum) has the same random status with the specified seed
        for param in curriculum_params:
            env.set_curriculum_params(param)
            envs.append(copy.deepcopy(env))
        curriculum_params.reverse()
        envs.reverse()

    envs.insert(0, None)

    labels = ["#sample", "reward", "reference reward"]
    if problem == "CSP":
        labels.append("log(Kappa)")
    buffers = np.zeros((len(labels), smooth_episode))
    save_buffers = np.zeros((len(labels), save_episode))

    for phase in range(len(curriculum_params)):
        warmup = (phase < len(curriculum_params) - 1)
        prefix = "warmup" if warmup else "final"
        sample_cnt = 0
        param = curriculum_params[phase]
        agent.set_curriculum_params(param)
        envs.pop(0)
        env = envs[0]

        if args.load_path is None:
            if sample_type == "pi^0":
                sampler = copy.deepcopy(agent)
            elif sample_type == "pi^t":
                sampler = agent
            else:
                if phase > 0:
                    sampler = copy.deepcopy(agent)
                else:
                    sampler = agent
        elif phase > 0:
            if sample_type == "pi^t":
                sampler = agent
            else:
                sampler = copy.deepcopy(agent)
        
        if not warmup and init_type == "pi^0":
            agent.clear_params()

        if problem == "CSP":
            pi_star = env.get_opt_policy()
            phi = agent.get_phi_all()

        for episode in range(phase_episode):
            reward = 0
            agent.zero_grad()
            for gstep in range(grad_cummu_step):
                reward += collect_data(env, sampler, agent)
            agent.update_param()

            buf_idx = episode % smooth_episode
            sample_cnt += grad_cummu_step * env.cnt_samples
            buffers[0, buf_idx] = sample_cnt
            reward /= grad_cummu_step
            buffers[1, buf_idx] = reward
            buffers[2, buf_idx] = env.reference()
            if problem == "CSP":
                buffers[3, buf_idx] = log(calc_kappa(env.probs, pi_star, agent.get_policy(), phi).cpu().numpy())
            save_buffers[:, episode % save_episode] = buffers[:, buf_idx]

            if (episode + 1) % smooth_episode == 0:
                logger.log_stat("%s %s" % (prefix, "episode"), episode + 1, sample_cnt)
                for i in range(1, len(labels)):
                    logger.log_stat("%s %s" % (prefix, labels[i]), buffers[i].mean(), sample_cnt)
                logger.print_recent_stats()

            if (episode + 1) % save_episode == 0:
                env.clean()
                ckpt_package = {"episode": episode + 1, "agent":agent, "envs":envs, "sampler":sampler}
                torch.save(ckpt_package, os.path.join(logdir, "checkpoint/%s/%08d.pt" % (prefix, episode + 1)))

                log_package = {"episode": episode + 1}
                for i in range(len(labels)):
                    log_package["%s %s" % (prefix, labels[i])] = save_buffers[i]
                torch.save(ckpt_package, os.path.join(logdir, "logdata/%s/%08d.pt" % (prefix, episode + 1)))

                logger.info("Saving to %s" % logdir)

                if problem == "CSP":
                    plot_prob_fig(agent, env, os.path.join(logdir, "result/%s/visualize%08d.jpg" % (prefix, episode + 1)), args.device)
