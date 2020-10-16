import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random 
from utils import *
import collections 

Experience=collections.namedtuple('Experience', field_names=[
	'state','action', 'reward', 'done', 'next_state'])

class DDQN:
	def __init__(self, env, gamma, epsilon, epi_num, buffer_size, batch_size, layer_num, hide_size,learning_rate):
		self.env = env 
		self.gamma = gamma 
		self.epsilon = epsilon
		self.epi_num = epi_num
		self.buffer_size = buffer_size 
		self.batch_size = batch_size 
		self.layer_num = layer_num 
		self.hide_size = hide_size
		self.learning_rate = learning_rate
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi_td_error = []
		self.init_net()
		self.init_replay_buffer()
		self._reset()

	def _reset():
		self.state = self.env.reset()
		self.total_reward = 0.0 
		self.step_num = 0 
		self.td_error = 0.0


	def init_net(self):
		self.net = DQN_1d(self.env.observation_space.shape[0], self.hide_size, self.env.action_space.n, self.layer_num)
		self.optimizer=optim.Adam(self.net.parameters(), lr=self.learning_rate)
		self.target_net = DQN_1d(self.env.observation_space.shape[0], self.hide_size, self.env.action_space.n, self.layer_num)
		self.loss_list = []

	def init_replay_buffer(self):
		self.buffer = ExperienceBuffer(self.buffer_size)

	def take_action(self):
		if random.uniform(0, 1) < self.epsilon:
			action = self.env.action_space.sample()
		else:			
			state_a=np.array([self.state], copy=False)
			state_v=torch.tensor(state_a).to(device)
			q_vals_v=self.net(state_v)
			_, act_v=torch.max(q_vals_v, dim=1)
			action=int(act_v.item()) # convert to int from tensor.

		next_state, reward, done, info = self.env.step(action)
		self.total_reward += reward 
		self.step_num += 1
		next_state_v = torch.tensor(np.array([next_state], copy=False)).to(device)
		next_q_val, _ = torch.max(net(next_state_v), dim=1)
		self.td_error = reward + self.gamma*float(next_q_val.item())-float(q_vals_v[act_v].item())

		exp = Experience(self.state, action, reward, done, next_state)
		self.buffer.append(exp)
		self.state = next_state

	def cal_loss(self, batch, double=True):
		'''
		double == True  means DDQN
		'''
		states, actions, rewards, dones, next_states=batch 
		states_v=torch.tensor(states).to(device)
		next_states_v=torch.tensor(next_states).to(device)
		actions_v=torch.tensor(actions).to(device)
		rewards_v=torch.tensor(rewards).to(device)
		done_mask=torch.ByteTensor(dones).to(device)

		state_action_values=self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
		if double == True:
			next_state_actions = self.net(next_states_v).max(1)[1]
			next_state_values = self.target_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
		else:
			next_state_values=self.target_net(next_states_v).max(1)[0]
		next_state_values[done_mask]=0.0 
		next_state_values=next_state_values.detach()
		expected_stata_action_values=rewards_v+next_state_values*gamma
		loss=nn.MSELoss()(state_action_values, expected_stata_action_values)
		return loss 

	def update_net(self):
		batch = self.buffer.sample(self.batch_size)
		loss = self.cal_loss(batch)
		self.loss_list.extend([loss])
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
	
	def update_target_net(self):
		self.targrt_net.load_state_dict(self.net.state_dict())

	def run(self):
		self.init_net()
		self.init_replay_buffer()
		for epi in range(epi_num):
			self._reset()
			done = False
			while not done:
				self.take_action()
				if len(self.buffer) > batch_size:
					if epi % 5 == 0:
						self.update_net()

			if epi % 10 == 0:
				self.update_target_net()

			if epi % 100 == 0:

				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				self.epi_td_error.extend([self.td_error])

				print('DDQN ~~~ epi {}, total reward {}, step num {}'.format(epi, total_reward, step_num))

		print('DDQN ~~ Training Finished !')
		return self.epi_total_reward, self.epi_step_num, self.epi_td_error,	self.loss_list







