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

Experience=collections.namedtuple('Experience', field_names=[
	'state','action', 'reward', 'done', 'next_state'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class DQN_node2vec:
	def __init__(self, env, size, state_num, gamma, epsilon, epi_num, max_step, learning_rate, buffer_size, batch_size, emb_dim, node2vec_matrix=None):
		self.env = env 
		self.node2vec_matrix = node2vec_matrix
		self.learned_embedding = None
		self.size = size
		self.state_num = state_num
		self.action_num = self.env.action_space.n
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
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi = 0
		self.loss_list = []
		self.init_net()
		self._reset()

	def _reset(self):
		self.env.reset()
		self.state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.done = False
		self.total_reward = 0.0 
		self.step_num = 0 
		self.traj = []

	def init_net(self):
		self.net = DQN_node2vec_nn(self.state_num, self.action_num, self.emb_dim, self.node2vec_matrix).to(device)
		self.optimizer=optim.Adam(self.net.parameters(), lr=self.learning_rate)
		

	def take_action(self):
		if self.node2vec_matrix is None:
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
		# self.state_matrix[self.state, self.state_neighbors[self.state]] =  1
		self.state_matrix[self.state, next_state] =  1
		np.fill_diagonal(self.state_matrix, 0)
		exp = Experience(self.state, action, reward, self.done, next_state)
		self.buffer.append(exp)
		self.total_reward += reward 
		self.step_num += 1
		self.traj.append(self.state)
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
		next_state_values = next_state_values.detach()
		expected_stata_action_values = rewards_v+next_state_values*self.gamma
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
		np.fill_diagonal(self.state_matrix, 1)
		g = nx.from_numpy_matrix(self.state_matrix)
		data = torch_geometric.utils.from_networkx(g)
		model = Node2Vec(data.edge_index, embedding_dim=self.emb_dim, walk_length=20,
				context_size=10, walks_per_node=10, num_negative_samples=1,
				p=1, q=1, sparse=True).to(device)
		loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
		emb_optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
		model.train()
		for epoch in range(200):
			total_loss = 0
			for pos_rw, neg_rw in loader:
				emb_optimizer.zero_grad()
				loss = model.loss(pos_rw.to(device), neg_rw.to(device))
				loss.backward()
				emb_optimizer.step()
				total_loss += loss.item()
			# print('learn node2vec embedding,  epoch %s, loss %s'%(epoch, total_loss / len(loader)))
		# print('data.num_nodes', data.num_nodes)
		# print('state_num', self.state_num)
		# learned_embedding = model(torch.arange(data.num_nodes, device=device))
		# self.learned_embedding = learned_embedding.cpu().detach()
		

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
			if epi % 1 == 0:
				print('node2vec ~~ epidose {}, total reward {}, step num {}'.format(epi, self.total_reward, self.step_num))
		if self.node2vec_matrix is None:
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Learn Node2vec Embedding ')
			self.learn_embedding()
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ node2vec Finished !')
		else:
			pass
		embedding = self.net.embedding.weight.cpu().detach()
		return self.epi_total_reward, embedding







