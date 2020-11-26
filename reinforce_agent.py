import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random 
from agent_utils import *
import collections 
import torch.nn.functional as F

Experience=collections.namedtuple('Experience', field_names=[
	'state','action', 'reward', 'done', 'next_state'])
device = 'cpu'

class REINFORCE:
	def __init__(self, env, gamma, epi_num, learning_rate):
		self.env = env 
		self.gamma = gamma 
		self.epi_num = epi_num 
		self.learning_rate = learning_rate
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi_loss = []
		self._reset()
		self.init_pg_net()

	def _reset(self):
		self.state = self.env.reset()
		self.done = False
		self.total_reward = 0.0 
		self.step_num = 0 
		self.episode_exp = []

	def init_pg_net(self):
		self.net = PG_net(self.env.observation_space.shape[0], self.env.action_space.n)
		self.optimizer = optim.Adam(self.net.parameters(), lr = self.learning_rate)

	def take_action(self):
		state_a = np.array([self.state], copy=False)
		state_v = torch.FloatTensor(state_a).to(device)
		logits_v = self.net(state_v)
		probs_v = F.softmax(logits_v, dim=1)
		probs_a = probs_v.data.cpu().numpy().flatten()
		action = np.random.choice(len(probs_a), p = probs_a)

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
		actions = []
		rewards = []
		q_vals = []
		for step, exp in enumerate(self.episode_exp):
			states.append(exp.state)
			actions.append(int(exp.action))
			rewards.append(exp.reward)
		# print('states', states)
		# print('actions', actions)
		# print('rewards', rewards)
		self.optimizer.zero_grad()
		q_vals.extend(cal_q_vals(rewards, self.gamma))
		# print('q_vals', q_vals)
		q_vals_v = torch.FloatTensor(q_vals)
		states_v = torch.FloatTensor(states)
		actions_v = torch.LongTensor(actions)
		logits_v = self.net(states_v)
		log_prob_v = F.log_softmax(logits_v, dim=1)
		# print('log_prob_v.shape', log_prob_v.size())

		# print('log_prob_v', log_prob_v)
		# print('log_prob_v[range(len(states)), actions]', log_prob_v[range(len(states)), actions])
		log_prob_actions_v = q_vals_v*log_prob_v[range(len(states_v)), actions_v]
		loss_v = -log_prob_actions_v.mean()
		loss_v.backward()
		self.optimizer.step()

		return loss_v.item()


	def run(self):
		for epi in range(self.epi_num):
			self._reset()
			while not self.done:
				self.take_action()
			loss = self.update_net() # update net at the end of each episode.
			if epi % 100 == 0:
				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				self.epi_loss.extend([loss])
				print('REINFORCE ~~~ epi {}, total_reward {}, step_num {}, los  {}'.format(epi, self.total_reward, self.step_num, loss))
		print('REINFORCE ~~~ Training Finished !')
		return self.epi_total_reward, self.epi_step_num, self.epi_loss











