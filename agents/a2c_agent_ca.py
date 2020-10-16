import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random 
import math
from utils import *
import collections 
import torch.nn.functional as F

Experience=collections.namedtuple('Experience', field_names=[
	'state','action', 'reward', 'done', 'next_state'])

class A2C_ca:
	'''
	A2C agent for continuous action space. 
	The dimenison of action feature is action size. 
	The actor output the mean and varinace of action feature. 
	Action can be selected by the mean vector or sampled from normal distribution N(mu, var). 
	'''
	def __init__(self, env, gamma,  epi_num, entropy_beta, learning_rate):
		self.env = env 
		self.gamma = gamma 
		self.epi_num = epi_num 
		self.learning_rate = learning_rate
		self.entropy_beta = entropy_beta 
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi_loss = []
		self._reset()
		self.init_net()

	def _reset(self):
		self.state = self.env.reset()
		self.total_reward = 0.0 
		self.step_num = 0
		self.loss = 0.0
		self.episode_exp = []

	def init_net(self):
		self.net = A2C_net_ca(self.env.observation_space.shape[0], self.env.action_space.shape[0])
		self.optimizer = optim.Adam(self.net.parameters(), lr = self.learning_rate)

	def take_action(self):
		state_a = np.array([self.state], copy=False)
		state_v = torch.tensor(state_a).to('cpu')
		mu_v, var_v, val_v = self.net(state_v)
		mu = mu_v.data.cpu().numpy()
		var = var_v.data.cpu().numpy()
		action = np.random.normal(mu,var)
		action = np.clip(action, -1, 1)
		next_state, reward, done, info = self.env.step(action)
		exp = Experience(self.state, action, reward, done, next_state)
		self.episode_exp.append(exp)

		self.total_reward += reward
		self.step_num += 1 
		self.state = next_stata

	def update_net(self):
		states = []
		next_states = []
		actions = []
		rewards = []
		for step, exp in enumerate(self.episode_exp):
			states.append(exp.state)
			next_states.append(exp.next_state)
			actions.append(int(exp.action))
			rewards.append(exp.reward)

		states_v = torch.FloatTensor(states)
		next_states_v = torch.FloatTensor(next_states)
		actions_v = torch.LongTensor(actions)
		rewards_v = torch.FloatTensor(rewards)

		self.optimizer.zero_grad()
		_, _, next_states_val_v = self.net(next_states_v)
		ref_vals_v = rewards_v + self.gamma * next_states_val_v

		mus_v, vars_v, states_val_v= self.net(states_v)
		loss_val_v = F.mse_loss(states_val_v, ref_vals_v)
		adv_v = ref_vals_v - states_val_v
		log_prob_v = adv_v * cal_logprob(mus_v, vars_v, actions_v)
		loss_policy_v = -log_prob_v.mean()
		loss_entropy_v = self.entropy_beta*(-(torch.log(2*math.pi*vars_v)+1)/2).mean()

		loss_v = loss_val_v + loss_entropy_v + loss_policy_v
		loss_v.backward()
		self.optimizer.step()
		self.epi_loss.extend([loss_v.data.cpu().numpy()])

	def  run(self):
		



