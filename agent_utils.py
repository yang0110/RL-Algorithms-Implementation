'''
This file contains some usefull block of codes: 
ExperienceBuffer, neural network
'''
import numpy as np 
import collections
import torch
import torch.nn as nn
import math 
import ptan 

def cal_adv_ref(trajectory, net_crt, states_v, device='cpu'):
    '''
    calculate advantage for training actor and reference values for training critic.

    parameter:
    ---------
    trajectory: several episodes concatenated together.

    return:
    ------
    advantages
    reference values
    '''
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
    '''
    Calculate q values from the end of episode 
    This is for REINFORCE algorithm

    parameters
    ==========
    Input: rewards of an episode 
    Ouput: the empirical return of each state-action pair (q value) of an episode
    ===========
    '''
    res = []
    sum_r = 0
    for r in reversed(rewards):
        sum_r *= gamma 
        sum_r += r 
        res.append(sum_r)

    return list(reversed(res))


# def cal_nstep_q_vals(rewards, n, gamma):
#     res = []
#     sum_r = 0
#     for r in reversed(rewards):
#         sum_r += gamma 
#         sum_r += r 
#         res.append(sum_r)
#     return list(reversed(res))



def cal_logprob(mu, var, actions):
    '''
    calculate log prob for A2C_agnet_ca
    '''
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
    """Update target net slowly

    parameters:
    -----
    alpha: step size

    return: 
    -------
    updated target net
    """
    assert isinstance(alpha, float)
    assert 0.0 < alpha <= 1.0
    state = net.state_dict()
    tgt_state = target_net.state_dict()
    for k, v in state.items():
        tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
    target_net.load_state_dict(tgt_state)
    return target_net

# class PrioritizedReplayBuffer(ExperienceReplayBuffer)


class DQN_conv(nn.Module):
    '''
    NN for input as 2-dimensional matrix: image
    '''
    def __init__(self, input_size, n_actions):
        super(DQN_conv, self).__init__()
        # Process the image input
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_size)
        # output the q value of each action
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)



class DQN_1d(nn.Module):
    '''
    DQN for input in the form of 1-dimensional vector
    '''
    def __init__(self, input_size, n_actions):
        super(DQN_1d, self).__init__()
        # Output the q value of each action
        self.net=nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
            )

    def forward(self, x):
        return self.net(x)


class DQN_dueling(nn.Module):
    '''
    Network for dueling DQN
    '''
    def __init__(self, input_size, n_actions):
        super(DQN_dueling, self).__init__()
        # Output the q value of each action
        self.net_adv=nn.Sequential(
            nn.Linear(input_size, hide_size),
            nn.ReLU(),
            nn.Linear(hide_size, n_actions)
            )
        self.net_val=nn.Sequential(
            nn.Linear(input_size, hide_size),
            nn.ReLU(),
            nn.Linear(hide_size, n_actions)
            )

    def forward(self, x):
        val = self.net_val(x)
        adv = self.net_adv(x)
        return val + adv - adv.mean()



class PG_net(nn.Module):
    '''
    Network for policy gradient algorithm. 

    Paramters:
    ==========
    Input: state featue 
    Output: Actions probability
    ==========
    '''
    def __init__(self, input_size,  n_actions):
        super(PG_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
            )

    def forward(self, x):
        x = self.net(x)
        return x



class A2C_net(nn.Module):
    def __init__(self, input_size, n_actions):
        super(A2C_net, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
            )

        self.value_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )

    def forward(self,x):
        action_prob = self.policy_net(x)
        state_value = self.value_net(x)
        return action_prob, state_value



class A2C_net_ca(nn.Module):
    '''
    Actot-critic net for continuous action (ca) space .

    Parameter
    ---------
    Input_size: observation dimension
    action_size: action feature diemension
    mu: return the mean of output action
    var: the variance of output action
    value: the state value
    '''
    def __init__(self, input_size, action_size):
        super(A2C_net_ca, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            )
        self.mu = nn.Sequential(
            nn.Linear(32, action_size),
            nn.Tanh(),
            )
        self.var = nn.Sequential(
            nn.Linear(32, action_size),
            nn.Softplus(),
            )
        self.value = nn.Linear(32, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


class DDPG_actor(nn.Module):
    '''
    return the action
    '''
    def __init__(self, input_size, action_size):
        super(DDPG_actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, action_size), 
            nn.Tanh(),
            )
    def forward(self, x):
        return self.net(x)


class DDPG_critic(nn.Module):
    '''
    return the q value
    '''
    def __init__(self, input_size, action_size):

        super(DDPG_critic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(input_size, 64), 
            nn.ReLU(),
            )
        self.out_net = nn.Sequential(
            nn.Linear(64 + action_size, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1),
            )
    def forward(self, x, a):
        obs = self.obs_net(x)
        q_val = self.out_net(torch.cat([obs, a], dim = 1))
        return q_val


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        # return the probabilityof actions 

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # return the estimated state value

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)



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






class ModelActor(nn.Module):
    '''
    The network is designed for continuous action

    parameters:
    ----------
    obs_size: observation dimension
    act_size: action vetor dimension

    return: 
    ------
    mean of action vector
    '''
    def __init__(self, obs_size,act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(), 
            np.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_size),
            nn.Tanh(),
            )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)

class ModelCritic(nn.Module):
    '''
    calculate the state value

    parameter:
    ---------
    obs_size: observation dimension

    return:
    ------
    state value (scaler)
    '''
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(obs_size, 64), 
            nn.ReLU(), 
            nn.Linear(64,64), 
            nn.ReLU(),
            nn.Linear(64, 1),
            )
    def forward(self, x):
        return self.value(x)

class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device = 'cpu'):
        self.net = net 
        self.device = device 

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        actions = mu + np.exp(logstd)* np.random.normal(size=logstd.shape)
        return actions, agent_states






















