import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random 
from agent_utils import *
from utils import *
from nn_models import *
import collections 
from torch_geometric.nn import Node2Vec
import torch_geometric
import networkx as nx
from scipy.sparse import csgraph

Experience=collections.namedtuple('Experience', field_names=[
	'state','action', 'reward', 'done', 'next_state'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class DQN_PVF:
	def __init__(self, env, size, state_num, beta, gamma, epsilon, epi_num, max_step,  learning_rate, buffer_size, batch_size, emb_dim, pvf_matrix=None):
		self.env = env 
		self.pvf_matrix = pvf_matrix
		self.size = size
		self.state_num = state_num
		self.action_num = self.env.action_space.n
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
		self.state_neighbors = {i: [0] for i in range(self.state_num)}
		self.state_matrix = np.zeros((self.state_num, self.state_num))
		self.state_count = np.zeros(self.state_num)
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi = 0
		self.loss_list = []
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
		self.net = DQN_pvf_nn(self.state_num, self.action_num, self.emb_dim, self.pvf_matrix).to(device)
		self.optimizer=optim.Adam(self.net.parameters(), lr=self.learning_rate)
		
	def take_action(self):
		if self.pvf_matrix is None:
			action = np.random.choice(range(self.action_num))
		else:
			self.epsilon = 0.9*0.99**self.epi + 0.1
			if random.uniform(0, 1) < self.epsilon:
				action = np.random.choice(range(self.action_num))
			else:			
				state_a=np.array([self.state], copy=False)
				state_v=torch.LongTensor(state_a).to(device)
				q_vals_v =self.net(state_v)
				_, act_v = torch.max(q_vals_v, dim=1)
				action = int(act_v.item()) 

		_, reward, self.done, info = self.env.step(action)
		next_state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.state_neighbors[self.state].append(next_state)
		self.state_neighbors[self.state] = np.unique(self.state_neighbors[self.state]).tolist()
		self.state_matrix[self.state, next_state] = 1
		self.total_reward += reward 
		self.step_num += 1
		self.state_count[self.state] += 1
		# reward += self.beta*(1/(np.sqrt(self.state_count[next_state]+1)*np.sqrt(len(self.state_neighbors[next_state])+1)))
		exp = Experience(self.state, action, reward, self.done, next_state)
		self.buffer.append(exp)
		self.state = next_state
		
	def cal_loss(self, batch, double=False):
		states, actions, rewards, dones, next_states = batch 
		states_v = torch.LongTensor(states).to(device)
		next_states_v = torch.LongTensor(next_states).to(device)
		actions_v = torch.LongTensor(actions).to(device)
		rewards_v = torch.FloatTensor(rewards).to(device)
		done_mask = torch.BoolTensor(dones).to(device)

		state_action_values =self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
		next_state_values = self.net(next_states_v).max(1)[0]
		next_state_values[done_mask] = 0.0 
		expected_stata_action_values = rewards_v + self.gamma*next_state_values.detach()
		loss = nn.MSELoss()(state_action_values, expected_stata_action_values)
		return loss

	def update_net(self):
		batch = self.buffer.sample(self.batch_size)
		self.optimizer.zero_grad()
		loss = self.cal_loss(batch)
		loss.backward()
		self.optimizer.step()
		return loss.item()
	
	def learn_embedding(self):
		adj = (self.state_matrix+self.state_matrix.T)/2
		lap = csgraph.laplacian(adj)
		eigval, eigvec = np.linalg.eig(lap)
		self.pvf_matrix = eigvec[:, :self.emb_dim]
		print('~~~~~~~~~~~~ ~~~~~~~ ~~~~~~~~~~~ Learn PVF')		

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
					if self.epi % 1 == 0:
						loss = self.update_net()
						self.loss_list.extend([loss])

			self.epi_total_reward.extend([self.total_reward])
			self.epi_step_num.extend([self.step_num])
			if self.epi % 1 == 0:
				print('PVF ~~ epidose {}, total reward {}, step num {}'.format(epi, self.total_reward, self.step_num))
		if self.pvf_matrix is None:
			self.learn_embedding()
		else:
			pass
		print('PVF ~~ Training Finished !')
		return self.epi_total_reward, torch.FloatTensor(self.pvf_matrix)







