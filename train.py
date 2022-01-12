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
import configparser
from logger import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--config", default="config.ini", type=str)
    parser.add_argument("--load-path", default=None, type=str)
    parser.add_argument("--override-phase-episode", default=None, type=int)
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

def simplify_path(path):
    s_list = path.split("/")
    temp_str = " ".join(s_list)
    s_list = temp_str.split()
    
    new_list = []
    
    for item in s_list:
        if item == ".":
            continue
            
        elif item == "..":
            if new_list:
                new_list.pop(-1)
                
        else:
            new_list.append(item)
            
    new_str = "/".join(new_list)
    if path[-1] == "/":
        new_str += "/"
    
    return new_str

def unpack_config(sample_type, init_type, seed, grad_cummu_step, phase_episode, save_episode, smooth_episode, **kwargs):
    return sample_type, init_type, int(seed), int(grad_cummu_step), int(phase_episode), int(save_episode), int(smooth_episode)

def unpack_config_csp(n_start, n_end, **kwargs):
    return [[int(n_start)], [int(n_end)]]

def unpack_config_olknd(n_start, n_end, B_start, B_end, V_start, V_end, **kwargs):
    return [[int(n_start), float(B_start), float(V_start)], [int(n_end), float(B_end), float(V_end)]]

def unpack_checkpoint(agent, envs, sampler, **kwargs):
    return agent, envs, sampler

def get_file_number(dir):
    _, fn = os.path.split(dir)
    num, _ = os.path.splitext(fn)
    num = "x" + num
    for i in range(len(num) - 1, -1, -1):
        if not num[i].isdigit():
            return int(num[i+1:])

def copy_logs(fn, logger, smooth_episode):
    log_package = torch.load(fn, map_location="cpu")
    for key, val in log_package.items():
        if "#sample" in key:
            sample_cnts = val
            break

    for key, val in log_package.items():
        if "episode" in key or "#sample" in key:
            continue
        save_episode = val.shape[0]
        for i in range(0, save_episode, smooth_episode):
            logger.log_stat(key, val[i:i+smooth_episode].mean(), int(sample_cnts[i+smooth_episode-1]))
    
    for key, val in log_package.items():
        if not "episode" in key:
            continue
        for i in range(0, save_episode, smooth_episode):
            logger.log_stat(key, val - save_episode + i + smooth_episode, int(sample_cnts[i+smooth_episode-1]))
    
    return int(sample_cnts[-1]) # returns the sample count

def collect_data(env, sampler, samp_a_unif, agent):
    env.new_instance()

    csiz = env.bs_per_horizon
    rewards = torch.zeros((env.bs,), dtype=torch.double, device=env.device)
    idx = torch.zeros_like(rewards)
    grads_logp = torch.zeros((env.bs, agent.d), dtype=torch.double, device=env.device)

    for i in range(env.horizon):
        il, ir = (env.horizon - (i + 1)) * csiz, (env.horizon - i) * csiz

        state = env.get_state()
        s_sampler, s_agent = state[:ir], state[il:]

        a_sampler, _ = sampler.get_action(s_sampler)
        a_agent, entropy = agent.get_action(s_agent)

        id = torch.randint(2, (csiz,), dtype=torch.double, device=env.device)
        unif = torch.randint_like(id, env.action_size)
        idx[il:ir] = id
        if samp_a_unif:
            a_sampler[-csiz:] = torch.randint_like(id, env.action_size)
        ax = id * a_sampler[-csiz:] + (1 - id) * a_agent[:csiz]
        
        action = torch.cat((a_sampler[:-csiz], ax, a_agent[csiz:]))
        reward, active = env.get_reward(action)

        log_prob, grad_logp = agent.query_sa(s_sampler[-csiz:], a_sampler[-csiz:])
        log_prob[log_prob < -agent.U] = -agent.U
        grads_logp[il:ir] = grad_logp * active[il:ir].unsqueeze(-1)
        
        rewards[il:ir] = active[il:ir] * (reward[il:ir] + agent.L * (id * (-log_prob[:csiz]) + (1 - id) * entropy[:csiz]))
        rewards[ir:] += reward[ir:] + agent.L * active[ir:] * entropy[csiz:]
    
    rewards = rewards * (4 * idx - 2)
    agent.store_grad(rewards, grads_logp)

def evaluate(env, agent):
    env.new_instance()
    rewards = torch.zeros((env.bs,), dtype=torch.double, device=env.device)
    sigma_t = torch.zeros((agent.d, agent.d), dtype=torch.double, device=env.device)
    for i in range(env.horizon):
        state = env.get_state()
        action, entropy = agent.get_action(state)
        reward, active = env.get_reward(action)
        rewards += reward
        
        unif = torch.randint_like(action, env.action_size)
        log_prob, grad_logp = agent.query_sa(state, unif)
        grad_logp = grad_logp * active.unsqueeze(-1)
        sigma_t += (grad_logp.T @ grad_logp) / grad_logp.shape[0]
    
    env.new_instance()
    ref_rewards = torch.zeros_like(rewards)
    sigma_star = torch.zeros_like(sigma_t)
    for i in range(env.horizon):
        state = env.get_state()
        action = env.get_reference_action()
        reward, active = env.get_reward(action)
        ref_rewards += reward

        log_prob, grad_logp = agent.query_sa(state, action)
        grad_logp = grad_logp * active.unsqueeze(-1)
        sigma_star += (grad_logp.T @ grad_logp) / grad_logp.shape[0]

    S, U = torch.symeig(sigma_t, eigenvectors=True)

    pos_eig = S > 0
    sqinv = 1 / S[pos_eig].sqrt()
    U = U[:, pos_eig]
    st = U @ torch.diag(sqinv) @ U.T

    e = torch.symeig(st @ sigma_star @ st.T)[0]
    return rewards.mean().item(), ref_rewards.mean().item(), log(e[-1])

if __name__ == "__main__":
    args = get_args()

    if args.load_path is not None:
        load_dir = simplify_path(os.path.join(os.path.dirname(args.load_path), "../../"))
        args.config = os.path.join(load_dir, "config.ini")

    parser = configparser.RawConfigParser()
    parser.optionxform = lambda option: option
    parser.read(args.config, encoding='utf-8')
    problem = dict(parser.items("Problem"))["name"]

    assert problem in ["CSP", "OLKnapsackDecision"]

    config = dict(parser.items(problem))
    sample_type, init_type, seed, grad_cummu_step, phase_episode, save_episode, smooth_episode = unpack_config(**config)
    if problem == "CSP":
        curriculum_params = unpack_config_csp(**config)
    elif problem == "OLKnapsackDecision":
        curriculum_params = unpack_config_olknd(**config)

    assert phase_episode % save_episode == 0
    assert save_episode % smooth_episode == 0
    assert sample_type in ["pi^0", "pi^t", "pi^t-pi^0"]
    assert init_type in ["pi^0", "pi^0-pi^t"]
    
    set_seed(seed)

    log_dir = "Exp-%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), problem)
    logger = Logger(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "code"), exist_ok=True)
    for par in ["checkpoint", "logdata", "result"]:
        for sub in ["", "/warmup", "/final"]:
            os.makedirs(os.path.join(log_dir, "%s%s" % (par, sub)), exist_ok=True)
    logger.info("Experiment dir: %s" % (log_dir))

    dirs = os.listdir(".")
    for fn in dirs:
        if os.path.splitext(fn)[-1] == ".py":
            shutil.copy(fn, os.path.join(log_dir, "code"))
    shutil.copy(args.config, os.path.join(log_dir, "config.ini"))

    cur_phase_episode = phase_episode

    if args.load_path is not None: # continue training
        logger.info("Migrating from %s." % (load_dir))
        is_final = "final" in args.load_path
        num = get_file_number(args.load_path)
        num = int(num)
        if is_final:
            limit = {"/warmup": 1e18, "/final": num}
            logger.info("warmup: all \t final: <= %d" % (num))
        else:
            limit = {"/warmup": num, "/final": -1}
            logger.info("warmup: <= %d \t final: none" % (num))
        
        st_sample_cnt = 0
        # copy files
        for par in ["checkpoint", "logdata", "result"]:
            for sub in ["/warmup", "/final"]:
                from_path = os.path.join(load_dir, "%s%s" % (par, sub))
                to_path = os.path.join(log_dir, "%s%s" % (par, sub))
                dir = os.listdir(from_path)
                dir.sort()
                lim = limit[sub]
                logger_output = "%s%s:" % (par, sub)
                for file in dir:
                    num = get_file_number(file)
                    if num <= lim:
                        logger_output += " " + file
                        fn = os.path.join(from_path, file)
                        shutil.copy(fn, to_path)
                        if par == "logdata":
                            _ = copy_logs(fn, logger, smooth_episode)
                            st_sample_cnt = max(st_sample_cnt, _)
                logger.info(logger_output)
        
        package = torch.load(args.load_path, map_location=args.device)
        agent, envs, sampler = unpack_checkpoint(**package)
        st_episode_num = package["episode"]
        not_reset = True

        logger.info("Done. Start training from episode %d with %d samples." % (st_episode_num, st_sample_cnt))
        
        if args.override_phase_episode is not None:
            cur_phase_episode = args.override_phase_episode
            logger.info("Override phase episode to %d." % (cur_phase_episode))
            assert cur_phase_episode % save_episode == 0

        for it in envs + [agent, sampler]:
            it.move_device(args.device)
        
        env_curriculum_params = envs[0].curriculum_params
        
        if is_final:
            curriculum_params = [curriculum_params[1]]
        else:
            curriculum_params = [env_curriculum_params, curriculum_params[1]]
    else: # initialize a new training
        if problem == "CSP":
            env = CSPEnv(args.device, **config)
            agent = CSPAgent(args.device, **config)
        elif problem == "OLKnapsackDecision":
            env = OLKnapsackDecisionEnv(args.device, **config)
            agent = OLKnapsackDecisionAgent(args.device, **config)
        
        envs = []

        if curriculum_params[0] == curriculum_params[1]:
            curriculum_params = [curriculum_params[0]]
        
        curriculum_params.reverse() # ensure that final environments (curriculum or not curriculum) has the same random status with the specified seed
        for param in curriculum_params:
            env.set_curriculum_params(param)
            envs.append(copy.deepcopy(env))
        curriculum_params.reverse()
        envs.reverse()

        st_sample_cnt, st_episode_num = 0, 0
        not_reset = False
    
    if len(curriculum_params) == 2:
        assert not (sample_type == "pi^t" and init_type == "pi^0"), "This will invalidate the first phase training!"

    if len(curriculum_params) == 1 and (sample_type == "pi^t-pi^0" or init_type == "pi^0-pi^t"):
        if args.load_path is None:
            if sample_type == "pi^t-pi^0":
                sample_type = "pi^t"
            if init_type == "pi^0-pi^t":
                init_type = "pi^0"
        else:
            if sample_type == "pi^t-pi^0":
                sample_type = "pi^0"
            if init_type == "pi^0-pi^t":
                init_type = "pi^t"
        logger.info("Only 1 phase training, but 2 phase designed for sample / init types: Override sample_type to %s, init_type to %s." % (sample_type, init_type))

    envs.insert(0, None)

    labels = ["#sample", "reward", "reference reward", "log(Kappa)"]
    buffers = torch.zeros((len(labels), smooth_episode))
    save_buffers = torch.zeros((len(labels), save_episode))

    for phase in range(len(curriculum_params)):
        warmup = (phase < len(curriculum_params) - 1)
        samp_a_unif = False if warmup else ()
        prefix = "warmup" if warmup else "final"
        sample_cnt = st_sample_cnt
        st_sample_cnt = 0
        param = curriculum_params[phase]
        agent.set_curriculum_params(param)
        envs.pop(0)
        env = envs[0]
        
        if sample_type == "pi^t-pi^0":
            _samp = "pi^t" if warmup else "pi^0"
        else:
            _samp = sample_type
        
        if _samp == "pi^0":
            if args.load_path is None or phase > 0:
                sampler = copy.deepcopy(agent)
            samp_a_unif = True
        else:
            sampler = agent
            samp_a_unif = False

        if not warmup and init_type == "pi^0" and not not_reset:
            agent.clear_params()
        not_reset = False

        episode_range = range(st_episode_num, st_episode_num + cur_phase_episode)
        st_episode_num = 0
        cur_phase_episode = phase_episode

        for episode in episode_range:
            agent.zero_grad()
            for gstep in range(grad_cummu_step):
                collect_data(env, sampler, samp_a_unif, agent)
            agent.update_param()
            reward, ref_reward, log_kappa = evaluate(env, agent)

            buf_idx = episode % smooth_episode
            sample_cnt += grad_cummu_step * env.cnt_samples
            buffers[0, buf_idx] = sample_cnt
            buffers[1, buf_idx] = reward
            buffers[2, buf_idx] = ref_reward
            buffers[3, buf_idx] = log_kappa
            save_buffers[:, episode % save_episode] = buffers[:, buf_idx]

            if (episode + 1) % smooth_episode == 0:
                logger.log_stat("%s %s" % (prefix, "episode"), episode + 1, sample_cnt)
                for i in range(1, len(labels)):
                    logger.log_stat("%s %s" % (prefix, labels[i]), buffers[i].mean(), sample_cnt)
                logger.print_recent_stats()

            if (episode + 1) % save_episode == 0:
                env.clean()
                ckpt_package = {"episode": episode + 1, "agent":agent, "envs":envs, "sampler":sampler}
                torch.save(ckpt_package, os.path.join(log_dir, "checkpoint/%s/%08d.pt" % (prefix, episode + 1)))

                log_package = {"%s episode" % (prefix): episode + 1}
                for i in range(len(labels)):
                    log_package["%s %s" % (prefix, labels[i])] = save_buffers[i]
                torch.save(log_package, os.path.join(log_dir, "logdata/%s/%08d.pt" % (prefix, episode + 1)))

                env.plot_prob_figure(agent, os.path.join(log_dir, "result/%s/%08d.jpg" % (prefix, episode + 1)))

                logger.info("Saving to %s" % log_dir)
