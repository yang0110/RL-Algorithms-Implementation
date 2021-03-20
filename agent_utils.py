'''
This file contains some usefull block of codes: 
ExperienceBuffer, neural network
'''
import numpy as np 
import collections
import torch
import torch.nn as nn
import math 

def cal_adv_ref(trajectory, net_crt, states_v, device='cpu'):
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
        if exp.done: 
            delta = exp.reward - val 
            last_gae = delta
        else:
            delta = exp.reward + gamma*next_val - val 
            last_gae = delta + gamma*gae_lambda*last_gae

        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v 



def cal_q_vals(rewards, gamma):
    res = []
    sum_r = 0
    for r in reversed(rewards):
        sum_r *= gamma 
        sum_r += r 
        res.append(sum_r)

    return list(reversed(res))


def cal_logprob(mu, var, actions):
    p1 = -((mu - actions)**2) / (2*var.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2*math.pi*var))
    return p1 + p2

class ExperienceBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for idx in indices:
            states.extend([self.buffer[idx].state])
            actions.extend([self.buffer[idx].action])
            rewards.extend([self.buffer[idx].reward])
            next_states.extend([self.buffer[idx].next_state])
            dones.extend([self.buffer[idx].done])

        states_a = np.array(states)
        actions_a = np.array(actions)
        rewards_a = np.array(rewards)
        next_states_a = np.array(next_states)
        dones_a = np.array(dones)

        states_v = torch.FloatTensor(states_a)
        actions_v = torch.LongTensor(actions_a)
        rewards_v = torch.FloatTensor(rewards_a)
        next_states_v = torch.FloatTensor(next_states_a)
        dones_v = torch.BoolTensor(dones_a)
        return states_v, actions_v, rewards_v, dones_v, next_states_v

class EpisodeBuffer:
    '''A buffer stores a list of episodes. Each episode constains a list of experience

        parameters:
        ----------
        buffer size: the maximum number of episode
        batch size: the number of episode to sample

        return: 
        ------
        The experience of sampled epsidoe in form of tensor
    '''
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for idx in indices:
            states.extend([self.buffer[idx].state])
            actions.extend([self.buffer[idx].action])
            rewards.extend([self.buffer[idx].reward])
            next_states.extend([self.buffer[idx].next_state])
            dones.extend([self.buffer[idx].done])

        states_a = np.array(states)
        actions_a = np.array(actions)
        rewards_a = np.array(rewards)
        next_states_a = np.array(next_states)
        dones_a = np.array(dones)

        states_v = torch.FloatTensor(states_a)
        actions_v = torch.FloatTensor(actions_a)
        rewards_v = torch.FloatTensor(rewards_a)
        next_states_v = torch.FloatTensor(next_states_a)
        dones_v = torch.BoolTensor(dones_a)
        return states_v, actions_v, rewards_v, dones_v, next_states_v


def alpha_sync(net, target_net, alpha):
    assert isinstance(alpha, float)
    assert 0.0 < alpha <= 1.0
    state = net.state_dict()
    tgt_state = target_net.state_dict()
    for k, v in state.items():
        tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
    target_net.load_state_dict(tgt_state)
    return target_net


def unpack_batch(batch, net, next_val_gamma, device='cpu'):
    '''
    convert batch into training tesnors

    parameters:
    ----------
    batch: transitions 
    net: agent net 

    return:
    -------
    states tensore, actions tensor
    '''
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    next_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state,copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.next_state is not None:
            not_done_idx.append(idx)
            next_states.append(np.array(exp.next_state, copy=False))
    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.LongTensor(actions).to(device)

    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        next_states_v = torch.FloatTensor(next_states).to(device)
        _, next_vals_v = next(next_states_v)
        next_vals_np = next_vals_v.data.cpu().numpy()[:,0]
        rewards_np[not_done_idx] += next_vals_np + next_val_gamma*next_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v 


























