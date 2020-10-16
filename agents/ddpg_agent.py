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

class DDPG:
	def __init__(self, env, gamma, epsilon, sigma, epi_num, buffer_size, batch_size, entropy_beta, learning_rate):
		self.env = env 
		self.gamma = gamma 
		self.epsilon = epsilon
		self.sigma = sigma
		self.epi_num = epi_num 
		self.buffer_size = buffer_size
		self.batch_size = batch_size
		self.buffer = ExperienceBuffer(self.buffer_size)
		self.entropy_beta = entropy_beta 
		self.learning_rate = learning_rate
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi_actor_loss = []
		self.epi_critic_loss = []
		self._reset()
		self.init_net()

	def _reset(self):
		self.state = self.env.reset()
		self.total_reward = 0.0 
		self.step_num = 0
		self.loss = 0.0
		self.episode_exp = []

	def init_net(self):
		self.actor_net = DDPG_actor(self.env.observation_space.shape[0], self.env.action_space.shape[0])
		self.critic_net = DDPG_critic(self.env.observation_space.shape[0], self.env.action_space.shape[0])
		self.target_actor_net = DDPG_actor(self.env.observation_space.shape[0], self.env.action_space.shape[0])
		self.target_critic_net = DDPG_critic(self.env.observation_space.shape[0], self.env.action_space.shape[0])

		self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = self.learning_rate)
		self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr = self.learning_rate)

	def take_action(self):
		state_a = np.array([self.state], copy=False)
		state_v = torch.tensor(state_a).to('cpu')
		action_v = self.actor_net(state_v)
		action  = action_v.data.cpu().numpy()
		if random.uniform(0, 1) < self.epsilon:
			action = action + self.epsilon * np.random.normal(size = self.env.action_space.shape[0], sigma = self.sigma)

		next_state, reward, done, info = self.env.step(action)
		exp = Experience(self.state, action, reward,  done, next_state)
		self.episode_exp.append(exp)
		self.buffer.append(exp)
		self.total_reward += reward 
		self.step_num += 1
		self.state = next_state

	def update_net(self):
		states_v, actions_v, rewards_v, dones_mask, next_states_v = self.buffer.sample(self.batch_size)
		self.optimizer_critic.zero_grad()
		q_v = self.critic_net(states_v, actions_v)
		next_actions_v = self.target_actor_net(states_v)
		next_q_v = self.target_critic_net(next_states_v, next_actions_v)
		next_q_v[dones_mask] = 0.0
		q_ref_v = rewards_v.unsqueeze(dim=1) + next_q_v * self.gamma
		self.critic_loss_v = F.mse_loss(q_v, next_q_v.detach())
		self.critic_loss_v.backward()
		self.optimizer_critic.step()

		self.optimizer_actor.zero_grad()
		actions_v = self.actor_net(states_v)
		self.actor_loss_v = - self.critic_net(states_v, actions_v).mean()
		self.actor_loss_v.backward()
		self.optimizer_actor.step()

	def update_target_net(self):
		self.targrt_actor_net.load_state_dict(self.actor_net.state_dict())
		self.targrt_critic_net.load_state_dict(self.critic_net.state_dict())

	def run(self):
		for epi in range(self.epi_num):
			self._reset()
			done = False
			while not done:
				self.take_action()
				if len(self.buffer) > self.batch_size:
					if epi % 5 == 0:
						self.update_net()

			if epi % 10 == 0:
				self.update_target_net()

			if epi % 100 == 0:

				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				self.epi_actor_loss.extend(self.actor_loss_v.data.cpu().numpy())
				self.epi_critic_loss.extend(self.critic_loss_v.data.cpu().numpy())

				print('DDPG ~~~ epi {}, total reward {}, step num {}'.format(epi, total_reward, step_num))

		print('DDPG ~~ Training Finished !')
		return self.epi_total_reward, self.epi_step_num, self.epi_actor_loss, self.epi_critic_loss


















