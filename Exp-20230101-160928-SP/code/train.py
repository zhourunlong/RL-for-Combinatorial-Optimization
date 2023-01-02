import argparse
import os
import sys
import time
import shutil
import numpy as np
import torch
import random
import time
import copy
import configparser
from math import *
from utils.logger import Logger
from utils.file_ops import *
from agents import REGISTRY as agents_REGISTRY
from envs import REGISTRY as envs_REGISTRY

unpack_config_REGISTRY = {}


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


def unpack_config(sample_type, init_type, seed, phase_episode, save_episode, smooth_episode, **kwargs):
    return sample_type, init_type, int(seed), int(phase_episode), int(save_episode), int(smooth_episode)


def unpack_config_SP(n_start, n_end, **kwargs):
    return [[int(n_start)], [int(n_end)]]


unpack_config_REGISTRY["SP"] = unpack_config_SP


def unpack_config_OKD(n_start, n_end, B_start, B_end, V_start, V_end, **kwargs):
    return [[int(n_start), float(B_start), float(V_start)], [int(n_end), float(B_end), float(V_end)]]


unpack_config_REGISTRY["OKD"] = unpack_config_OKD


def unpack_checkpoint(agent, envs, sampler, **kwargs):
    return agent, envs, sampler


def collect_data(env, sampler, samp_a_unif, agent):
    env.new_instance()

    csiz = env.bs_per_horizon
    rewards = torch.zeros((env.bs,), dtype=torch.double, device=env.device)
    idx = torch.zeros_like(rewards)
    grads_logp = torch.zeros(
        (env.bs, agent.d), dtype=torch.double, device=env.device)
    sigma_t = torch.zeros((agent.d, agent.d),
                          dtype=torch.double, device=env.device)

    for i in range(env.horizon):
        il, ir = -(i + 2) * csiz, -(i + 1) * csiz

        state = env.get_state()
        s_sampler, s_agent = state[:ir], state[il:]

        a_sampler, _ = sampler.get_action(s_sampler)
        a_agent, entropy = agent.get_action(s_agent)

        id = torch.randint(2, (csiz,), dtype=torch.double, device=env.device)
        idx[il:ir] = id
        if samp_a_unif:
            a_sampler[-csiz:] = torch.randint_like(id, env.action_size)
        ax = id * a_sampler[-csiz:] + (1 - id) * a_agent[:csiz]

        action = torch.cat((a_sampler[:-csiz], ax, a_agent[csiz:]))
        print(action)
        reward, active = env.get_reward(action)

        log_prob, grad_logp = agent.query_sa(s_sampler, a_sampler)
        log_prob[log_prob < -agent.U] = -agent.U
        grad_logp *= active[:ir].unsqueeze(-1)
        grads_logp[il:ir] = grad_logp[-csiz:]
        sigma_t += (grad_logp.T @ grad_logp) / grad_logp.shape[0]

        rewards[il:ir] = active[il:ir] * (reward[il:ir] + agent.L * (
            id * (-log_prob[:csiz]) + (1 - id) * entropy[:csiz]))
        rewards[ir:-csiz] += reward[ir:-csiz] + agent.L * \
            active[ir:-csiz] * entropy[csiz:-csiz]
        rewards[-csiz:] += reward[-csiz:]

    rewards[:-csiz] *= 4 * idx[:-csiz] - 2
    agent.store_grad(rewards, grads_logp)
    return rewards[-csiz:].mean().item(), sigma_t


def evaluate(env, agent, g_t):
    env.new_instance()

    csiz = env.bs_per_horizon
    rewards = torch.zeros((env.bs,), dtype=torch.double, device=env.device)
    idx = torch.zeros_like(rewards)
    grads_logp = torch.zeros(
        (env.bs, agent.d), dtype=torch.double, device=env.device)
    sigma_star = torch.zeros(
        (agent.d, agent.d), dtype=torch.double, device=env.device)

    for i in range(env.horizon):
        il, ir = (env.horizon - i) * csiz, (env.horizon + 1 - i) * csiz

        state = env.get_state()
        s_agent = state[il:]

        a_env = env.get_reference_action()[:ir]
        a_agent, entropy = agent.get_action(s_agent)

        id = torch.randint(2, (csiz,), dtype=torch.double, device=env.device)
        idx[il:ir] = id
        ax = id * a_env[-csiz:] + (1 - id) * a_agent[:csiz]

        action = torch.cat((a_env[:-csiz], ax, a_agent[csiz:]))
        reward, active = env.get_reward(action)

        log_prob, grad_logp = agent.query_sa(state[:ir], a_env)
        log_prob[log_prob < -agent.U] = -agent.U
        grad_logp *= active[:ir].unsqueeze(-1)
        grads_logp[il:ir] = grad_logp[-csiz:]
        sigma_star += (grad_logp.T @ grad_logp) / grad_logp.shape[0]

        rewards[il:ir] = active[il:ir] * (reward[il:ir] + agent.L * (
            id * (-log_prob[:csiz]) + (1 - id) * entropy[:csiz]))
        rewards[ir:-csiz] += reward[ir:-csiz] + agent.L * \
            active[ir:-csiz] * entropy[csiz:-csiz]
        rewards[:csiz] += reward[:csiz]

    rewards[csiz:] *= 4 * idx[csiz:] - 2
    err_t = (rewards[csiz:].unsqueeze(-1) -
             grads_logp[csiz:] @ g_t).sum().item() / csiz
    return rewards[:csiz].mean().item(), sigma_star, err_t


def calc_log_kappa(sigma_t, sigma_star):
    S, U = torch.symeig(sigma_t, eigenvectors=True)

    pos_eig = S > 0
    sqinv = 1 / S[pos_eig].sqrt()
    U = U[:, pos_eig]
    st = U @ torch.diag(sqinv) @ U.T

    e = torch.symeig(st @ sigma_star @ st.T)[0]
    return log(e[-1])


if __name__ == "__main__":
    args = get_args()

    if args.load_path is not None:
        load_dir = simplify_path(os.path.join(
            os.path.dirname(args.load_path), "../../"))
        args.config = os.path.join(load_dir, "config.ini")

    parser = configparser.RawConfigParser()
    parser.optionxform = lambda option: option
    parser.read(args.config, encoding='utf-8')
    problem = dict(parser.items("Problem"))["name"]

    assert problem in ["SP", "OKD"]

    config = dict(parser.items(problem))
    sample_type, init_type, seed, phase_episode, save_episode, smooth_episode = unpack_config(
        **config)
    curriculum_params = unpack_config_REGISTRY[problem](**config)
# SP: N_start, n_end

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
            os.makedirs(os.path.join(log_dir, "%s%s" %
                        (par, sub)), exist_ok=True)
    logger.info("Experiment dir: %s" % (log_dir))

    dirs = os.listdir(".")
    for fn in dirs:
        if os.path.splitext(fn)[-1] == ".py":
            shutil.copy(fn, os.path.join(log_dir, "code"))
    shutil.copy(args.config, os.path.join(log_dir, "config.ini"))

    cur_phase_episode = phase_episode

    if args.load_path is not None:  # continue training
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

        logger.info("Done. Start training from episode %d with %d samples." % (
            st_episode_num, st_sample_cnt))

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
    else:  # initialize a new training
        env = envs_REGISTRY[problem](args.device, **config)
        agent = agents_REGISTRY[problem](args.device, **config)

        envs = []

        if curriculum_params[0] == curriculum_params[1]:
            curriculum_params = [curriculum_params[0]]

        # ensure that final environments (curriculum or not curriculum) has the same random status with the specified seed
        curriculum_params.reverse()
        for param in curriculum_params:
            env.set_curriculum_params(param)
            envs.append(copy.deepcopy(env))
        curriculum_params.reverse()
        envs.reverse()

        st_sample_cnt, st_episode_num = 0, 0
        not_reset = False

    if len(curriculum_params) == 2:
        assert not (sample_type == "pi^t" and init_type ==
                    "pi^0"), "This will invalidate the first phase training!"

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
        logger.info("Only 1 phase training, but 2 phase designed for sample / init types: Override sample_type to %s, init_type to %s." %
                    (sample_type, init_type))

    envs.insert(0, None)

    labels = ["#sample", "reward", "reference reward", "err_t", "log(Kappa)"]
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

        episode_range = range(
            st_episode_num, st_episode_num + cur_phase_episode)
        st_episode_num = 0
        cur_phase_episode = phase_episode

        for episode in episode_range:
            agent.zero_grad()
            reward, sigma_t = collect_data(env, sampler, samp_a_unif, agent)
            g_t = agent.update_param()
            ref_reward, sigma_star, err_t = evaluate(env, agent, g_t)
            log_kappa = calc_log_kappa(sigma_t, sigma_star)

            buf_idx = episode % smooth_episode
            sample_cnt += env.cnt_samples
            save_buffers[:, episode % save_episode] = buffers[:, buf_idx] = torch.tensor(
                [sample_cnt, reward, ref_reward, err_t, log_kappa])

            if (episode + 1) % smooth_episode == 0:
                logger.log_stat("%s %s" % (prefix, "episode"),
                                episode + 1, sample_cnt)
                for i in range(1, len(labels)):
                    logger.log_stat("%s %s" % (
                        prefix, labels[i]), buffers[i].mean(), sample_cnt)
                logger.print_recent_stats()

            if (episode + 1) % save_episode == 0:
                env.clean()
                ckpt_package = {"episode": episode + 1,
                                "agent": agent, "envs": envs, "sampler": sampler}
                torch.save(ckpt_package, os.path.join(
                    log_dir, "checkpoint/%s/%08d.pt" % (prefix, episode + 1)))

                log_package = {"%s episode" % (prefix): episode + 1}
                for i in range(len(labels)):
                    log_package["%s %s" %
                                (prefix, labels[i])] = save_buffers[i]
                torch.save(log_package, os.path.join(
                    log_dir, "logdata/%s/%08d.pt" % (prefix, episode + 1)))

                env.plot_prob_figure(agent, os.path.join(
                    log_dir, "result/%s/%08d.jpg" % (prefix, episode + 1)))

                logger.info("Saving to %s" % log_dir)
