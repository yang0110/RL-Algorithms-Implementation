import numpy as np 
import collections
import torch
import torch.nn as nn
import math 
import ptan 

'''
THis file contains neural networks models of RL algorithms: 
DQN, REINFORCE, Actor-Critic, A2C, A3C, PPO, DDPG, TD3, SAC

'''

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

class DistributionalDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DistributionalDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        return fc_out.view(batch_size, -1, N_ATOMS)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())

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




