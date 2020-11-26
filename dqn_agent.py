import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random 
from agent_utils import *
import collections 

Experience=collections.namedtuple('Experience', field_names=[
	'state','action', 'reward', 'done', 'next_state'])

class DQN:
	def __init__(self, env, gamma, epsilon, epi_num, learning_rate, buffer_size, batch_size):
		self.env = env 
		self.gamma = gamma 
		self.epsilon = epsilon
		self.epi_num = epi_num
		self.buffer_size = buffer_size 
		self.buffer = ExperienceBuffer(self.buffer_size)
		self.batch_size = batch_size 
		self.learning_rate = learning_rate
		self.epi_total_reward = []
		self.epi_step_num = []
		self.init_net()
		self._reset()

	def _reset(self):
		self.state = self.env.reset()
		self.done = False
		self.total_reward = 0.0 
		self.step_num = 0 


	def init_net(self):
		self.net = DQN_1d(self.env.observation_space.shape[0], self.env.action_space.n)
		self.optimizer=optim.Adam(self.net.parameters(), lr=self.learning_rate)
		self.target_net = DQN_1d(self.env.observation_space.shape[0], self.env.action_space.n)
		self.loss_list = [0]

	def take_action(self):
		if random.uniform(0, 1) < self.epsilon:
			action = self.env.action_space.sample()
		else:			
			state_a=np.array([self.state], copy=False)
			state_v=torch.FloatTensor(state_a).to('cpu')
			q_vals_v=self.net(state_v)
			_, act_v=torch.max(q_vals_v, dim=1)
			action=int(act_v.item()) # convert to int from tensor.

		next_state, reward, self.done, info = self.env.step(action)
		self.total_reward += reward 
		self.step_num += 1
		next_state_v = torch.FloatTensor(np.array([next_state], copy=False)).to('cpu')
		next_q_val, _ = torch.max(self.net(next_state_v), dim=1)
		exp = Experience(self.state, action, reward, self.done, next_state)
		self.buffer.append(exp)
		self.state = next_state

	def cal_loss(self, batch, double=True):
		'''
		double == True  means DDQN
		'''
		states, actions, rewards, dones, next_states=batch 
		states_v=torch.FloatTensor(states).to('cpu')
		next_states_v=torch.FloatTensor(next_states).to('cpu')
		actions_v=torch.LongTensor(actions).to('cpu')
		rewards_v=torch.FloatTensor(rewards).to('cpu')
		done_mask=torch.BoolTensor(dones).to('cpu')

		state_action_values=self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
		if double == True:
			next_state_actions = self.net(next_states_v).max(1)[1]
			next_state_values = self.target_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
		else:
			next_state_values=self.target_net(next_states_v).max(1)[0]
		next_state_values[done_mask]=0.0 
		next_state_values=next_state_values.detach()
		expected_stata_action_values=rewards_v+next_state_values*self.gamma
		loss=nn.MSELoss()(state_action_values, expected_stata_action_values)
		return loss 

	def update_net(self):
		batch = self.buffer.sample(self.batch_size)
		self.optimizer.zero_grad()
		loss = self.cal_loss(batch)
		loss.backward()
		self.optimizer.step()
		return loss.item()
	
	def update_target_net(self):
		self.target_net.load_state_dict(self.net.state_dict())

	def run(self):
		self.init_net()
		for epi in range(self.epi_num):
			self._reset()
			while not self.done:
				self.take_action()
				if len(self.buffer) > self.batch_size:
					if epi % 5 == 0:
						loss = self.update_net()
						self.loss_list.extend([loss])

			if epi % 10 == 0:
				self.update_target_net()

			if epi % 100 == 0:

				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				print('DQN ~~~ epi {}, total reward {}, step num {}, loss {}'.format(epi, self.total_reward, self.step_num, self.loss_list[-1]))

		print('DQN ~~ Training Finished !')
		return self.epi_total_reward, self.epi_step_num, self.loss_list







