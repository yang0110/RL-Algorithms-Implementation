'''
This file contains some usefull block of codes: 
ExperienceBuffer, neural network
'''
import numpy as np 
import collections
import torch
import torch.nn as nn
import math 

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



def cal_logprob(mu, var, actions):
    '''
    calculate log prob for A2C_agnet_ca
    '''
    p1 = -((mu - actions)**2) / (2*var.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2*math.pi*var))
    return p1 + p2

class ExperienceBuffer:
	def __init__(self, buffer_size):
		self.buffer=collections.deque(maxlen=buffer_size)

	def __len__(self):
		return len(self.buffer)

	def append(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		indices=np.random.choice(len(self.buffer), batch_size, replace=False)
		states, actions, rewards, dones, next_states=zip(*[self.buffer[idx] for idx in indices])
		return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(states)


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
            nn.Linear(input_size, hide_size),
            nn.ReLU(),
            nn.Linear(hide_size, n_actions)
            )

    def forward(self, x):
        return self.net(x.float())


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



class PG_net(n.Module):
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
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
            )

        def forward(self, x):
            return self.net(x)



class A2C_net(nn.Module):
    def __init__(self, input_size, n_actions):
        super(A2C_net, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
            )

        self.value_net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            )

        def forward(self,x):
            action_prob = self.policy_net(x)
            state_value = self.value_net(x)
            return action_prob, state_value



class A2C_net_ca(nn.Module):
    '''
    Actot-critic net for continuous action (ca) space .

    Parameter:
    =========
    Input_size: observation dimension
    action_size: action feature diemension
    mu: return the mean of output action
    var: the variance of output action
    value: the state value
    '''
    def __init__(self, input_size, action_size):
        super(A2C_net_ca, self).__init__():
        self.base = nn.Sequential(
            nn.Linear(input_size, hide_size),
            nn.ReLU(),
            )
        self.mu = nn.Sequential(
            nn.Linear(hide_size, action_size),
            n.Tanh(),
            )
        self.var = nn.Sequential(
            nn.Linear(hide_size, action_size),
            nn.Softplus(),
            )
        self.value = nn.Linear(hide_size, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


class DDPG_actor(nn.Module):
    '''
    return the action
    '''
    def __init__(self, input_size, action_size):
        super(DDPG_actor, self).__init__():
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), 
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_size), 
            nn.Tanh(),
            )
    def forward(self, x):
        return self.net(x)


class DDPG_critic(nn.Module):
    '''
    return the q value
    '''
    def __init__(self, input_size, action_size):
        super(DDPG_critic, self)__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(input_size, 128), 
            nn.ReLU(),
            )
        self.out_net = nn.Sequential(
            nn.Linear(128 + action_size, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1),
            )
    def forward(self, x, a):
        obs = self.obs_net(x)
        q_val = self.out_net(torch.cat([obs, a]), dim = 1)
        return q_val







