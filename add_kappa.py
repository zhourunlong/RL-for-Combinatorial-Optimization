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
    parser.add_argument("--load-path", default=None, type=str)
    return parser.parse_args()

def unpack_checkpoint(agent, envs, sampler, **kwargs):
    return agent, envs, sampler

def calc_log_kappa(env, agent):
    env.new_instance()
    grads_logp = torch.zeros((env.horizon, env.bs, agent.d), dtype=torch.double, device=env.device)
    for i in range(env.horizon):
        state = env.get_state()
        action, entropy = agent.get_action(state)
        reward, active = env.get_reward(action)

        unif = torch.randint_like(action, 2)
        log_prob, grad_logp = agent.query_sa(state, unif)
        grads_logp[i] = grad_logp * active.unsqueeze(-1)

    grads_logp = grads_logp.view(-1, agent.d)
    sigma_t = (grads_logp.T @ grads_logp) / grads_logp.shape[0]
    del grads_logp
    
    env.new_instance()
    env.reference()
    r = env.r
    grads_logp = torch.zeros((env.horizon, env.bs, agent.d), dtype=torch.double, device=env.device)
    for i in range(env.horizon):
        state = env.get_state()
        action = (state[:, 0] < r * state[:, 1]).double()
        reward, active = env.get_reward(action)

        log_prob, grad_logp = agent.query_sa(state, action)
        grads_logp[i] = grad_logp * active.unsqueeze(-1)
    
    grads_logp = grads_logp.view(-1, agent.d)
    sigma_star = (grads_logp.T @ grads_logp) / grads_logp.shape[0]
    del grads_logp

    S, U = torch.symeig(sigma_t, eigenvectors=True)

    pos_eig = S > 0
    sqinv = 1 / S[pos_eig].sqrt()
    U = U[:, pos_eig]
    st = U @ torch.diag(sqinv) @ U.T

    e = torch.symeig(st @ sigma_star @ st.T)[0]
    return log(e[-1])

if __name__ == "__main__":
    args = get_args()

    assert args.load_path is not None
    load_dir = args.load_path
    args.config = os.path.join(load_dir, "config.ini")
    
    st_sample_cnt = 0
    for sub in ["warmup", "final"]:
        ckpt_path = os.path.join(load_dir, "checkpoint/%s" % (sub))
        log_path = os.path.join(load_dir, "logdata/%s" % (sub))
        dir = os.listdir(ckpt_path)
        dir.sort()
        for file in dir:
            ckpt_fn = os.path.join(ckpt_path, file)
            print("%s/%s" % (sub, file))
            ckpt_package = torch.load(ckpt_fn, map_location=args.device)
            agent, envs, sampler = unpack_checkpoint(**ckpt_package)
            env = envs[0]
            agent.move_device(args.device)
            env.move_device(args.device)

            log_kappa = calc_log_kappa(env, agent)
            print(log_kappa)

            log_fn = os.path.join(log_path, file)
            log_package = torch.load(log_fn, map_location=args.device)
            log_package["%s log(Kappa)" % (sub)] = torch.full_like(log_package["%s reward" % (sub)], log_kappa)
