import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random 
import math
from agent_utils import *
from nn_modules import *
import collections 
import torch.nn.functional as F

Experience=collections.namedtuple('Experience', field_names=[
	'state','action', 'reward', 'done', 'next_state'])

class SAC:
	def __init__(self, env, gamma, epsilon, sigma, alpha, epi_num, buffer_size, batch_size, learning_rate_actor, learning_rate_critic):
		self.env = env 
		self.gamma = gamma 
		self.alpha = alpha
		self.epsilon = epsilon
		self.sigma = sigma
		self.epi_num = epi_num 
		self.buffer_size = buffer_size
		self.batch_size = batch_size
		self.exp_buffer = ExperienceBuffer(self.buffer_size)
		self.learning_rate_actor = learning_rate_actor
		self.learning_rate_critic = learning_rate_critic
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi_actor_loss = [0]
		self.epi_critic_loss = [0]
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
		self.actor_net = sac_actor(self.env.observation_space.shape[0], self.env.action_space.shape[0])
		self.critic_net = sac_critic(self.env.observation_space.shape[0], self.env.action_space.shape[0])

		self.target_actor_net = sac_actor(self.env.observation_space.shape[0], self.env.action_space.shape[0])
		self.target_critic_net = sac_critic(self.env.observation_space.shape[0], self.env.action_space.shape[0])

		self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = self.learning_rate_actor)
		self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr = self.learning_rate_critic)

	def take_action(self):
		state_a = np.array([self.state], copy=False)
		state_v = torch.FloatTensor(state_a).to('cpu')
		action_v, log_std = self.actor_net(state_v)
		action_v = action_v + log_std.exp()*torch.randn_like(action_v)
		action_v = torch.tanh(action_v).to('cpu')
		action  = action_v.squeeze(dim=-1).data.cpu().numpy()
		next_state, reward, self.done, info = self.env.step(action)
		exp = Experience(self.state, action, reward,  self.done, next_state)
		self.exp_buffer.append(exp)
		self.total_reward += reward 
		self.step_num += 1
		self.state = next_state

	def update_net(self):

	def update_target_net(self):
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net.load_state_dict(self.critic_net.state_dict())

	def run(self):
		for epi in range(self.epi_num):
			self._reset()
			while not self.done:
				self.take_action()
			if len(self.exp_buffer) > self.batch_size and epi % 10 == 0:
				actor_loss, critic_loss = self.update_net()
				self.epi_actor_loss.extend([actor_loss])
				self.epi_critic_loss.extend([critic_loss])
			
			if epi % 20 == 0:
				self.update_target_net()
				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				print('SAC ~~~ epi {}, total reward {}, step num {}, actor loss {}, critic loss {}'.format(epi, self.total_reward, self.step_num, self.epi_actor_loss[-1], self.epi_critic_loss[-1]))

		print('SAC ~~ Training Finished !')
		return self.epi_total_reward, self.epi_step_num, self.epi_actor_loss, self.epi_critic_loss


















