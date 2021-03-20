import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random 
from utils import *
from nn_models import *
import collections 

Experience=collections.namedtuple('Experience', field_names=[
	'state','action', 'reward', 'done', 'next_state'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class State2Emb:
	def __init__(self, env, size, state_num, beta, gamma, epsilon, epi_num, max_step, learning_rate, buffer_size, batch_size, emb_dim, state2emb_matrix=None, state_matrix=None):
		self.env = env 
		self.state2emb_matrix = state2emb_matrix
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
		if state_matrix is None:
			self.state_matrix = -np.ones((self.state_num, self.state_num))
		else:
			self.state_matrix = state_matrix
		self.state_neighbors = {i: [] for i in range(self.state_num)}
		self.total_state_neighbors = {i: [] for i in range(self.state_num)}
		self.state_count = [0 for i in range(self.state_num)]
		self.states = []
		self.observed_stat_num = []
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi = 0
		self.loss1_list = []
		self.loss2_list = []
		self.trans_loss_list = []
		self.class_loss = 0.0
		self.traj = []
		self.all_traj = {}
		self.init_net()
		self._reset()

	def _reset(self):
		self.env.reset()
		self.state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.state_neighbors = {i: [] for i in range(self.state_num)}
		self.traj = []
		self.done = False
		self.total_reward = 0.0 
		self.step_num = 0 

	def init_net(self):
		self.emb_net = State2emb_embedding_nn(self.state_num, self.action_num, self.emb_dim, self.state2emb_matrix).to(device)
		self.q_net = State2emb_q_nn(self.action_num, self.emb_dim).to(device)
		self.emb_optimizer = optim.Adam(self.emb_net.parameters(), lr= self.learning_rate)
		self.q_optimizer = optim.Adam(self.q_net.parameters(), lr= self.learning_rate)

	def take_action(self):
		self.epsilon = 0.9*0.99**self.epi + 0.1
		if random.uniform(0, 1) < self.epsilon:
			action = np.random.choice(range(self.action_num))
		else:			
			state_a=np.array([self.state], copy=False)
			state_v=torch.LongTensor(state_a).to(device)
			state_emb, _ = self.emb_net(state_v)
			q_vals_v =self.q_net(state_emb.detach())
			_, act_v = torch.max(q_vals_v, dim=1)
			action = int(act_v.item()) 

		_, reward, self.done, info = self.env.step(action)
		next_state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		if self.state not in self.states:
			self.states.append(self.state) 
		self.observed_stat_num.append(len(self.states))
		self.state_neighbors[self.state].append(next_state)
		self.state_neighbors[self.state] = np.unique(self.state_neighbors[self.state]).tolist()
		self.state_count[self.state] += 1
		# self.total_state_neighbors[self.state].append(next_state)
		# self.total_state_neighbors[self.state] = np.unique(self.total_state_neighbors[self.state]).tolist()
		# self.state_matrix[self.state, self.total_state_neighbors[self.state]] =  1
		self.state_matrix[self.state, next_state] = 1
		np.fill_diagonal(self.state_matrix, 0)
		self.total_reward += reward 
		self.step_num += 1
		self.traj.append(self.state)
		# reward += self.beta*(1/(np.sqrt(self.state_count[next_state]+1)*np.sqrt(len(self.state_neighbors[next_state])+1)))
		exp = Experience(self.state, action, reward, self.done, next_state)
		self.buffer.append(exp)
		self.state = next_state

	def update_emb_net(self):
		# if self.state2emb_matrix is None:
		sample_states = np.random.choice(range(self.state_num), size=self.batch_size)
		sample_states_v = torch.LongTensor(sample_states).to(device)
		batch_state_matrix = self.state_matrix[sample_states][:, sample_states]
		batch_state_matrix = torch.FloatTensor(batch_state_matrix).to(device)
		embs_v, matrix = self.emb_net(sample_states_v)
		self.emb_optimizer.zero_grad()
		class_loss = nn.L1Loss()(torch.reshape(matrix, (-1,)), torch.reshape(batch_state_matrix, (-1,)))
		class_loss.backward()
		self.emb_optimizer.step()
		self.class_loss = class_loss.item()
		# else:
			# pass
		return self.class_loss

	def update_q_net(self):
		batch = self.buffer.sample(self.batch_size)
		states, actions, rewards, dones, next_states = batch 
		states_v = torch.LongTensor(states).to(device)
		next_states_v = torch.LongTensor(next_states).to(device)
		actions_v = torch.LongTensor(actions).to(device)
		rewards_v = torch.FloatTensor(rewards).to(device)
		done_mask = torch.BoolTensor(dones).to(device)
		states_emb, matrix =self.emb_net(states_v)
		state_action_values = self.q_net(states_emb.detach())
		state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
		next_states_emb, _ = self.emb_net(next_states_v)
		next_state_values = self.q_net(next_states_emb.detach()).max(1)[0]
		next_state_values[done_mask] = 0.0 
		next_state_values = next_state_values.detach()
		expected_stata_action_values = rewards_v+next_state_values*self.gamma
		self.q_optimizer.zero_grad()
		loss = nn.MSELoss()(state_action_values, expected_stata_action_values)
		loss.backward()
		self.q_optimizer.step()
		return loss.item()
	
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
					if self.step_num % 1 == 0:
						q_loss = self.update_q_net()
						self.loss1_list.extend([q_loss])

					if self.step_num % 5 == 0:
						cla_loss = self.update_emb_net()
						self.loss2_list.extend([cla_loss])

			self.epi_total_reward.extend([self.total_reward])
			self.epi_step_num.extend([self.step_num])
			if epi % 1 == 0:
				print('State2emb ~~ epidose {}, total reward {}, step num {}'.format(epi, self.total_reward, self.step_num))
			if (epi <= 10) or (epi>=self.epi_num-10):
				self.all_traj[epi] = self.traj
		print('State2Emb ~~ Training Finished !')
		embedding = self.emb_net.embedding.weight.cpu().detach()
		return self.epi_total_reward, embedding, self.state_matrix, self.state_count, self.all_traj, self.observed_stat_num, self.state_matrix







