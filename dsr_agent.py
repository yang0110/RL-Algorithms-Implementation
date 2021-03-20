import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random 
from agent_utils import *
from utils import *
from nn_models import *
import collections 


Experience=collections.namedtuple('Experience', field_names=[
	'state','action', 'reward', 'done', 'next_state'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DSR:
	def __init__(self, env, size, state_num, beta, gamma, epsilon, epi_num, max_step, learning_rate, buffer_size, batch_size, emb_dim, dsr_matrix=None):
		self.env = env 
		self.state_num = state_num
		self.action_num = self.env.action_space.n
		self.size = size
		self.dsr_matrix = dsr_matrix
		self.beta = beta
		self.gamma = gamma 
		self.epsilon = epsilon
		self.epi_num = epi_num
		self.max_step = max_step
		self.buffer_size = buffer_size 
		self.buffer = ExperienceBuffer(self.buffer_size)
		self.batch_size = batch_size 
		self.learning_rate = learning_rate
		self.emb_dim = emb_dim
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi = 0
		self.state_count = np.zeros(self.state_num)
		self.reward_loss_list = []
		self.sr_loss_list = []
		self.init_net()
		self._reset()

	def _reset(self):
		self.env.reset()
		self.state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.done = False
		self.total_reward = 0.0 
		self.step_num = 0 

	def init_net(self):
		self.emb_net = DSR_embedding_nn(self.state_num, self.action_num, self.emb_dim, self.dsr_matrix).to(device)
		self.sr_net = DSR_sr_nn(self.emb_dim)
		self.emb_optimizer=optim.Adam(self.emb_net.parameters(), lr=self.learning_rate)
		self.sr_optimizer = optim.Adam(self.sr_net.parameters(), lr=self.learning_rate)

	def take_action(self):
		self.epsilon = 0.9*0.99**self.epi + 0.1
		if random.uniform(0, 1) < self.epsilon:
			action = np.random.choice(range(self.action_num))
		else:			
			state_a=np.array([self.state], copy=False)
			state_v=torch.LongTensor(state_a).to(device)
			_, state_emb_v = self.emb_net(state_v)
			sr_v = self.sr_net(state_emb_v)
			q_vals, _ = self.emb_net(state_v, sr=sr_v)
			action = np.argmax(q_vals.cpu().detach().numpy())

		_, reward, self.done, info = self.env.step(action)
		next_state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		next_state_v = torch.LongTensor(np.array([next_state], copy=False)).to(device)
		self.total_reward += reward 
		self.step_num += 1
		self.state_count[self.state] += 1
		exp = Experience(self.state, action, reward, self.done, next_state)
		self.buffer.append(exp)
		self.state = next_state

	def update_emb_net(self):
		states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)
		states_v = torch.LongTensor(states).to(device)
		next_states_v = torch.LongTensor(next_states).to(device)
		actions_v = torch.LongTensor(actions).to(device)
		rewards_v = torch.FloatTensor(rewards).to(device)
		done_mask = torch.BoolTensor(dones).to(device)

		est_rewards_v, states_emb_v = self.emb_net(states_v)
		est_rewards_v = est_rewards_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
		self.emb_optimizer.zero_grad()
		reward_loss = nn.MSELoss()(est_rewards_v, rewards_v)
		reward_loss.backward()
		self.emb_optimizer.step()
		return reward_loss.item()

	def update_sr_net(self):
		states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)
		states_v = torch.LongTensor(states).to(device)
		next_states_v = torch.LongTensor(next_states).to(device)
		actions_v = torch.LongTensor(actions).to(device)
		rewards_v = torch.FloatTensor(rewards).to(device)
		done_mask = torch.BoolTensor(dones).to(device)

		_, states_emb_v = self.emb_net(states_v)
		_, next_states_emb_v = self.emb_net(next_states_v)
		sr_v = self.sr_net(states_emb_v)
		next_sr_v = self.sr_net(next_states_emb_v)
		self.sr_optimizer.zero_grad()
		sr_loss = nn.MSELoss()(states_emb_v+self.gamma*next_sr_v, sr_v)
		sr_loss.backward()
		self.sr_optimizer.step()
		return sr_loss.item()

	def run(self):
		self.init_net()
		for epi in range(self.epi_num):
			self.epi = epi
			self._reset()
			while not self.done:
				self.take_action()
				if self.step_num > self.max_step:
					break 
				if len(self.buffer) > self.batch_size:
					if self.step_num % 10 == 0:
						reward_loss = self.update_emb_net()
						sr_loss = self.update_sr_net()
						self.reward_loss_list.append(reward_loss)
						self.sr_loss_list.append(sr_loss)

			self.epi_total_reward.extend([self.total_reward])
			self.epi_step_num.extend([self.step_num])
			if epi % 1 == 0:
				print('DSR ~~ epidose {}, total reward {}, step num {}'.format(epi, self.total_reward, self.step_num))

		print('DSR ~~ Training Finished !')
		embedding = self.emb_net.embedding.weight.cpu().detach()
		return self.epi_total_reward, embedding, self.state_count, self.reward_loss_list, self.sr_loss_list






