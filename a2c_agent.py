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

class A2C:
	'''
	Actor-critic + advantage value
	'''
	def __init__(self, env, gamma, epi_num, learning_rate, entropy_beta):
		self.env = env 
		self.gamma = gamma 
		self.epi_num = epi_num 
		self.learning_rate = learning_rate
		self.entropy_beta = entropy_beta
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi_loss = []
		self.init_a2c_net()
		self._reset()

	def _reset(self):
		self.state = self.env.reset()
		self.done = False
		self.total_reward = 0.0 
		self.step_num = 0
		self.episode_exp = []

	def init_a2c_net(self):
		self.net = A2C_net(self.env.observation_space.shape[0], self.env.action_space.n)
		self.optimizer = optim.Adam(self.net.parameters(), lr = self.learning_rate)

	def take_action(self):
		state_a = np.array([self.state], copy=False)
		state_v = torch.FloatTensor(state_a).to('cpu')
		logits_v, _ = self.net(state_v)
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
		_, next_states_val_v = self.net(next_states_v)
		ref_vals_v = torch.reshape(rewards_v, (len(rewards_v),1)) + self.gamma*next_states_val_v
		# print('rewards_v.size', rewards_v.size())
		# print('next_states_val_v.size', next_states_val_v.size())
		logits_v, states_val_v= self.net(states_v)
		loss_val_v = F.mse_loss(states_val_v, ref_vals_v)
		# print('states_val_v.size', states_val_v.size())
		# print('ref_vals_v.size', ref_vals_v.size())

		log_prob_v = F.log_softmax(logits_v, dim=1)
		adv_v = ref_vals_v - states_val_v
		log_prob_actions_v = adv_v*log_prob_v[range(len(states_v)), actions_v]
		loss_policy_v = -log_prob_actions_v.mean()

		prov_v = F.softmax(logits_v, dim=1)
		loss_entropy_v = self.entropy_beta*(prov_v*log_prob_v).sum(dim=1).mean()

		loss_policy_v.backward(retain_graph=True)
		loss_v = loss_val_v + loss_entropy_v
		loss_v.backward()
		self.optimizer.step()
		loss = loss_val_v + loss_policy_v + loss_entropy_v

		return loss.item()

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
				print('A2C ~~~ epi {}, total_reward {}, step_num {}, loss {}'.format(epi, self.total_reward, self.step_num, loss))
		print('A2C ~~~ Training Finished !')
		return self.epi_total_reward, self.epi_step_num, self.epi_loss			


