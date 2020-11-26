import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random 
import math
from agent_utils import *
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
		self.done = False
		self.total_reward = 0.0 
		self.step_num = 0
		self.loss = 0.0
		self.episode_exp = []

	def init_net(self):
		self.net = A2C_net_ca(self.env.observation_space.shape[0], self.env.action_space.shape[0])
		self.optimizer = optim.Adam(self.net.parameters(), lr = self.learning_rate)

	def take_action(self):
		state_a = np.array(self.state, copy=False)
		# print('state_a', state_a)
		# print('state_a.shape', state_a.shape)
		state_v = torch.FloatTensor(state_a).to('cpu')
		# print('state_v.shape', state_v.size())
		mu_v, var_v, val_v = self.net(state_v)
		mu = mu_v.data.cpu().numpy()
		var = var_v.data.cpu().numpy()
		action = np.random.normal(mu, var)
		action = np.clip(action, -1, 1)
		next_state, reward, self.done, info = self.env.step(action)
		exp = Experience(self.state, action, reward, self.done, next_state)
		self.episode_exp.append(exp)

		self.total_reward += reward
		self.step_num += 1 
		self.state = next_state

	def update_net(self):
		'''
		Update the policy at the end of each eposide
		'''
		states = []
		next_states = []
		actions = []
		rewards = []
		for step, exp in enumerate(self.episode_exp):
			states.append(exp.state)
			next_states.append(exp.next_state)
			actions.append(int(exp.action))
			rewards.append([exp.reward])

		states_v = torch.FloatTensor(states)
		next_states_v = torch.FloatTensor(next_states)
		actions_v = torch.LongTensor(actions)
		rewards_v = torch.FloatTensor(rewards)
		# print('states_v,size', states_v.size())

		self.optimizer.zero_grad()
		_, _, next_states_val_v = self.net(next_states_v)
		ref_vals_v = rewards_v + self.gamma * next_states_val_v
		# print('rewards_v.size', rewards_v.size())
		# print('next_states_val_v.size()', next_states_val_v.size())
		# print('ref_vals_v.size', ref_vals_v.size())
		mus_v, vars_v, states_val_v= self.net(states_v)
		# print('states_val_v.size', states_val_v.size())
		loss_val_v = F.mse_loss(states_val_v, ref_vals_v)
		adv_v = ref_vals_v - states_val_v
		log_prob_v = adv_v * cal_logprob(mus_v, vars_v, actions_v)
		loss_policy_v = -log_prob_v.mean()
		loss_entropy_v = self.entropy_beta*(-(torch.log(2*math.pi*vars_v)+1)/2).mean()

		loss_v = loss_val_v + loss_entropy_v + loss_policy_v
		loss_v.backward()
		self.optimizer.step()
		self.epi_loss.extend([loss_v.data.cpu().numpy()])
		return loss_v.data.cpu().numpy()

	def  run(self):
		for epi in range(self.epi_num):
			self._reset()
			while not self.done:
				self.take_action()
			loss = self.update_net() # update net at the end of each episode.
			if epi % 100 == 0:
				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				print('A2C ca ~~~ epi {}, total_reward {}, step_num {}, loss {}'.format(epi, self.total_reward, self.step_num, loss))
		print('A2C ca ~~~ Training Finished !')
		return self.epi_total_reward, self.epi_step_num, self.epi_loss	


