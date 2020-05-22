import math
import os
import datetime
import time
import pathlib
import random

import torch
import numpy as np
from collections import deque

def get_initial_state(env, state_buffer, action_repeat):
    state = env.reset(False)
    states = np.stack([state for _ in range(action_repeat)], axis=1)
    state_buffer = deque(maxlen=action_repeat)
    for _ in range(action_repeat):
        state_buffer.append(state)
    return states, state_buffer

def add_state(state, state_buffer):
    state_buffer.append(state)
    return state_buffer

def recent_state(state_buffer):
    state = np.array(state_buffer)
    return np.transpose(state, (1, 0, 2, 3))

def del_record(index, state_buffer, states_deque, actions_deque, rewards_deque):
    for i in range(len(state_buffer)):
        state_buffer[i] = np.delete(state_buffer[i], index, axis=0)
    if states_deque is not None:
        del states_deque[index]
        del actions_deque[index]
        del rewards_deque[index]
    return state_buffer, states_deque, actions_deque, rewards_deque

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def epsilon_scheduler(eps_start, eps_final, eps_decay):
    def function(frame_idx):
        return eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)
    return function

def beta_scheduler(beta_start, beta_frames):
    def function(frame_idx):
        return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    return function

def create_log_dir(args):
    log_dir = ""
    if args.multi_step != 1:
        log_dir = log_dir + "{}-step-".format(args.multi_step)
    if args.c51:
        log_dir = log_dir + "c51-"
    if args.prioritized_replay:
        log_dir = log_dir + "per-"
    if args.dueling:
        log_dir = log_dir + "dueling-"
    if args.double:
        log_dir = log_dir + "double-"
    if args.noisy:
        log_dir = log_dir + "noisy-"
    log_dir = log_dir + "dqn-"
    
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = log_dir + now

    log_dir = os.path.join("runs", log_dir)
    return log_dir

def print_log(frame, prev_frame, prev_time, reward_list, length_list, loss_list):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    avg_reward = np.mean(reward_list)
    avg_length = np.mean(length_list)
    avg_loss = np.mean(loss_list) if len(loss_list) != 0 else 0.

    print("Frame: {:<8} FPS: {:.2f} Avg. Reward: {:.2f} Avg. Length: {:.2f} Avg. Loss: {:.2f}".format(
        frame, fps, avg_reward, avg_length, avg_loss
    ))

def print_args(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

def save_model(model, args):
    fname = ""
    if args.multi_step != 1:
        fname += "{}-step-".format(args.multi_step)
    if args.c51:
        fname += "c51-"
    if args.prioritized_replay:
        fname += "per-"
    if args.dueling:
        fname += "dueling-"
    if args.double:
        fname += "double-"
    if args.noisy:
        fname += "noisy-"
    fname += "dqn-{}.pth".format(args.save_model)
    fname = os.path.join("models", fname)

    pathlib.Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), fname)

def load_model(model, args):
    if args.load_model is not None:
        fname = os.path.join("models", args.load_model)
    else:
        fname = ""
        if args.multi_step != 1:
            fname += "{}-step-".format(args.multi_step)
        if args.c51:
            fname += "c51-"
        if args.prioritized_replay:
            fname += "per-"
        if args.dueling:
            fname += "dueling-"
        if args.double:
            fname += "double-"
        if args.noisy:
            fname += "noisy-"
        fname += "dqn-{}.pth".format(args.save_model)
        fname = os.path.join("models", fname)

    if args.device == torch.device("cpu"):
        map_location = lambda storage, loc: storage
    else:
        map_location = None
    
    if not os.path.exists(fname):
        raise ValueError("No model saved with name {}".format(fname))

    model.load_state_dict(torch.load(fname, map_location))

def set_global_seeds(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)
