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

class DSF:
	def __init__(self, env, size, state_num, beta, gamma, epsilon, epi_num, max_step, learning_rate, buffer_size, batch_size, emb_dim, dsf_matrix=None):
		self.env = env 
		self.state_num = state_num
		self.action_num = self.env.action_space.n
		self.size = size
		self.dsf_matrix = dsf_matrix
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
		self.state_neighbors = {i: [] for i in range(self.state_num)}
		self.sf_loss_list = []
		self.q_loss_list = []
		self.init_net()
		self._reset()

	def _reset(self):
		self.env.reset()
		self.state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.state_neighbors = {i: [] for i in range(self.state_num)}
		self.done = False
		self.total_reward = 0.0 
		self.step_num = 0 

	def init_net(self):
		self.sf_net = DSF_sf_nn(self.state_num, self.emb_dim, self.dsf_matrix).to(device)
		self.q_net = DSF_q_nn(self.action_num, self.emb_dim)
		self.sf_optimizer = optim.Adam(self.sf_net.parameters(), lr=self.learning_rate)
		self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

	def take_action(self):
		self.epsilon = 0.9*0.99**self.epi + 0.1
		if random.uniform(0, 1) < self.epsilon:
			action = np.random.choice(range(self.action_num))
		else:			
			state_a=np.array([self.state], copy=False)
			state_v=torch.LongTensor(state_a).to(device)
			_, state_sf = self.sf_net(state_v)
			q_vals = self.q_net(state_sf)
			action = np.argmax(q_vals.cpu().detach().numpy())

		_, reward, self.done, info = self.env.step(action)
		next_state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.total_reward += reward 
		self.step_num += 1
		# self.state_count[self.state] += 1
		# self.state_neighbors[self.state].append(next_state)
		# self.state_neighbors[self.state] = np.unique(self.state_neighbors[self.state]).tolist()
		# reward += self.beta*(1/(np.sqrt(self.state_count[next_state]+1)*np.sqrt(len(self.state_neighbors[next_state])+1)))
		exp = Experience(self.state, action, reward, self.done, next_state)
		self.buffer.append(exp)
		self.state = next_state

	def update_sf_net(self):
		sf_loss = torch.FloatTensor([0.0])
		if self.dsf_matrix is None:
			states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)
			states_v = torch.LongTensor(states).to(device)
			next_states_v = torch.LongTensor(next_states).to(device)
			actions_v = torch.LongTensor(actions).to(device)
			rewards_v = torch.FloatTensor(rewards).to(device)
			done_mask = torch.BoolTensor(dones).to(device)

			state_embs, state_sfs = self.sf_net(states_v)
			_, next_state_sfs = self.sf_net(next_states_v)
			self.sf_optimizer.zero_grad()
			sf_loss = nn.MSELoss()(state_embs+self.gamma*next_state_sfs.detach(), state_sfs)
			sf_loss.backward()
			self.sf_optimizer.step()
		else:
			pass
		return sf_loss.item()

	def update_q_net(self):
		states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)
		states_v = torch.LongTensor(states).to(device)
		next_states_v = torch.LongTensor(next_states).to(device)
		actions_v = torch.LongTensor(actions).to(device)
		rewards_v = torch.FloatTensor(rewards).to(device)
		done_mask = torch.BoolTensor(dones).to(device)

		_, state_sfs = self.sf_net(states_v)
		_, next_state_sfs = self.sf_net(next_states_v)
		est_q = self.q_net(state_sfs).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
		est_next_q = self.q_net(next_state_sfs).max(1)[0]
		est_next_q[done_mask] = 0.0 
		est_next_q = est_next_q.detach()
		target = rewards_v+self.gamma*est_next_q
		self.q_optimizer.zero_grad()
		q_loss = nn.MSELoss()(target, est_q)
		q_loss.backward()
		self.q_optimizer.step()
		return q_loss.item()

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
					if self.step_num % 1 ==0:
						q_loss = self.update_q_net()
						self.q_loss_list.append(q_loss)
					if self.step_num % 5 == 0:
						sf_loss = self.update_sf_net()
						self.sf_loss_list.append(sf_loss)

			self.epi_total_reward.extend([self.total_reward])
			self.epi_step_num.extend([self.step_num])
			if epi % 1 == 0:
				print('DSF ~~ epidose {}, total reward {}, step num {}'.format(epi, self.total_reward, self.step_num))

		print('DSF ~~ Training Finished !')
		_, state_sfs = self.sf_net(torch.LongTensor(range(self.state_num)))
		return self.epi_total_reward, state_sfs, self.state_count, self.sf_loss_list, self.q_loss_list






