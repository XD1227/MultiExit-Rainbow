import torch
import torch.optim as optim

import os
from collections import deque
from common.utils import load_model, get_initial_state, add_state, recent_state, del_record
from model import DQN
from env import PanicEnv

def test(env, args): 
    current_model = DQN(env, args).to(args.device)
    current_model.eval()

    load_model(current_model, args)

    episode_reward = 0
    episode_length = 0

    state_buffer = deque(maxlen=args.action_repeat)
    states_deque = actions_deque = rewards_deque = None
    state, state_buffer = get_initial_state(env, state_buffer, args.action_repeat)
    while True:

        action = current_model.act(torch.FloatTensor(state).to(args.device), 0.)
        next_state, _, done, end = env.step(action, save_screenshots=True)
        add_state(next_state, state_buffer)
        next_state = recent_state(state_buffer)

        state = next_state

        if end:
            break
        # delete the agents that have reached the goal
        r_index = 0
        for r in range(len(done)):
            if done[r] is True:
                state_buffer, states_deque, actions_deque, rewards_deque = \
                    del_record(r_index, state_buffer, states_deque, actions_deque, rewards_deque)
                r_index -= 1
            r_index += 1
        next_state = recent_state(state_buffer)

        state = next_state
    PanicEnv.display(True)
    print("Test Result - Reward {} Length {}".format(episode_reward, episode_length))
    