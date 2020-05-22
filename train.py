import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import numpy as np
from env import PanicEnv, Radius
from collections import deque

from common.utils import get_initial_state, add_state, recent_state, del_record, epsilon_scheduler, beta_scheduler, update_target, print_log, load_model, save_model
from model import DQN
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

def train(env, args, writer): 
    current_model = DQN(env, args).to(args.device)
    target_model = DQN(env, args).to(args.device)

    if args.noisy:
        current_model.update_noisy_modules()
        target_model.update_noisy_modules()

    if args.load_model:  # and os.path.isfile(args.load_model)
        load_model(current_model, args)
        load_model(target_model, args)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)
    beta_by_frame = beta_scheduler(args.beta_start, args.beta_frames)

    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.alpha)
    else:
        replay_buffer = ReplayBuffer(args.buffer_size)
    
    state_buffer = deque(maxlen=args.action_repeat)
    states_deque = [deque(maxlen=args.multi_step) for _ in range(args.num_agents)] 
    rewards_deque = [deque(maxlen=args.multi_step) for _ in range(args.num_agents)] 
    actions_deque = [deque(maxlen=args.multi_step) for _ in range(args.num_agents)] 

    optimizer = optim.Adam(current_model.parameters(), lr=args.lr)

    reward_list, length_list, loss_list = [], [], []
    episode_reward = 0
    episode_length = 0
    episode = 0

    prev_time = time.time()
    prev_frame = 1

    state, state_buffer = get_initial_state(env, state_buffer, args.action_repeat)
    for frame_idx in range(1, args.max_frames + 1):

        if args.noisy:
            current_model.sample_noise()
            target_model.sample_noise()

        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)

        next_state, reward, done, end = env.step(action, save_screenshots=False)
        add_state(next_state, state_buffer)
        next_state = recent_state(state_buffer)

        for agent_index in range(len(done)):
            states_deque[agent_index].append((state[agent_index]))
            rewards_deque[agent_index].append(reward[agent_index])
            actions_deque[agent_index].append(action[agent_index])
            if len(states_deque[agent_index]) == args.multi_step or done[agent_index]:
                n_reward = multi_step_reward(rewards_deque[agent_index], args.gamma)
                n_state = states_deque[agent_index][0] 
                n_action = actions_deque[agent_index][0]
                replay_buffer.push(n_state, n_action, n_reward, next_state[agent_index], np.float32(done[agent_index]))
        
        # delete the agents that have reached the goal
        r_index = 0
        for r in range(len(done)):
            if done[r] is True:
                state_buffer, states_deque, actions_deque, rewards_deque = del_record(r_index, state_buffer, states_deque, actions_deque, rewards_deque)
                r_index -= 1
            r_index += 1
        next_state = recent_state(state_buffer)
        
        state = next_state
        episode_reward += np.array(reward).mean()
        episode_length += 1

        if end:
            if args.save_video and episode % 10 == 0:
                evaluate(env, current_model, args)   
            state, state_buffer = get_initial_state(env, state_buffer, args.action_repeat)
            reward_list.append(episode_reward)
            length_list.append(episode_length)
            writer.add_scalar("data/episode_reward", episode_reward, frame_idx)
            writer.add_scalar("data/episode_length", episode_length, frame_idx)
            episode_reward, episode_length = 0, 0
            for d in range(len(states_deque)):
                states_deque[d].clear()
                rewards_deque[d].clear()
                actions_deque[d].clear()
            states_deque = [deque(maxlen=args.multi_step) for _ in range(args.num_agents)] 
            rewards_deque = [deque(maxlen=args.multi_step) for _ in range(args.num_agents)] 
            actions_deque = [deque(maxlen=args.multi_step) for _ in range(args.num_agents)] 
            episode += 1 

        if len(replay_buffer) > args.learning_start and frame_idx % args.train_freq == 0:
            beta = beta_by_frame(frame_idx)
            losses = 0
            for _ in range(1):             
                loss = compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta)
                losses += loss.item()
            loss_list.append(losses)
            writer.add_scalar("data/loss", loss.item(), frame_idx)

        if frame_idx % args.update_target == 0:
            update_target(current_model, target_model)

        if frame_idx % args.evaluation_interval == 0:
            print_log(frame_idx, prev_frame, prev_time, reward_list, length_list, loss_list)
            reward_list.clear(), length_list.clear(), loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()
            save_model(current_model, args)

    save_model(current_model, args)

def compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta=None):
    """
    Calculate loss and optimize for non-c51 algorithm
    """
    if args.prioritized_replay:
        state, action, reward, next_state, done, weights, indices = replay_buffer.sample(args.batch_size, beta)
    else:
        state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
        weights = torch.ones(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    if not args.c51:
        q_values = current_model(state)
        target_next_q_values = target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        if args.double:
            next_q_values = current_model(next_state)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)
        else:
            next_q_value = target_next_q_values.max(1)[0]

        expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-5
        loss = (loss * weights).mean()
    
    else:
        q_dist = current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(args.batch_size, 1, args.num_atoms)
        q_dist = q_dist.gather(1, action).squeeze(1)
        q_dist.data.clamp_(0.01, 0.99)

        target_dist = projection_distribution(current_model, target_model, next_state, reward, done, 
                                              target_model.support, target_model.offset, args)

        loss = - (target_dist * q_dist.log()).sum(1)
        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-6
        loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    if args.prioritized_replay:
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()

    return loss


def projection_distribution(current_model, target_model, next_state, reward, done, support, offset, args):
    delta_z = float(args.Vmax - args.Vmin) / (args.num_atoms - 1)

    target_next_q_dist = target_model(next_state)

    if args.double:
        next_q_dist = current_model(next_state)
        next_action = (next_q_dist * support).sum(2).max(1)[1]
    else:
        next_action = (target_next_q_dist * support).sum(2).max(1)[1]

    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(target_next_q_dist.size(0), 1, target_next_q_dist.size(2))
    target_next_q_dist = target_next_q_dist.gather(1, next_action).squeeze(1)

    reward = reward.unsqueeze(1).expand_as(target_next_q_dist)
    done = done.unsqueeze(1).expand_as(target_next_q_dist)
    support = support.unsqueeze(0).expand_as(target_next_q_dist)

    Tz = reward + args.gamma * support * (1 - done)
    Tz = Tz.clamp(min=args.Vmin, max=args.Vmax)
    b = (Tz - args.Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    target_dist = target_next_q_dist.clone().zero_()
    target_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_next_q_dist * (u.float() - b)).view(-1))
    target_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_next_q_dist * (b - l.float())).view(-1))

    return target_dist

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret
def evaluate(env, curr_model, args):
    curr_model.eval()

    state_buffer = deque(maxlen=args.action_repeat)
    states_deque = actions_deque = rewards_deque = None
    state, state_buffer = get_initial_state(env, state_buffer, args.action_repeat)
    while True:
        action = curr_model.act(torch.FloatTensor(state).to(args.device), 0.)
        next_state, _, done, end = env.step(action, save_screenshots=True)
        add_state(next_state, state_buffer)
        next_state = recent_state(state_buffer)

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
    curr_model.train()
