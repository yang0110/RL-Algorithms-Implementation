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

class SF:
	def __init__(self, env, size, state_num, dim, gamma, epsilon, epi_num, max_step, learning_rate, state_f=None):
		self.env = env
		self.size = size
		self.state_num = state_num
		self.action_num = self.env.action_space.n
		self.dim = dim
		self.gamma = gamma 
		self.epsilon = epsilon
		self.epi_num = epi_num
		self.max_step = max_step
		self.epi = 0
		self.learning_rate = learning_rate
		self.q_table = np.zeros((self.state_num, self.action_num))
		self.state_f = state_f
		self.sr_matrix = np.zeros((self.state_num, self.state_num))
		self.w = np.zeros(self.dim)
		self.traj = []
		self.all_traj = {}
		self.state_count = np.zeros(self.state_num)
		self.epi_total_reward = []
		self.epi_step_num = []
		self.state_neighbors = {i: [] for i in range(self.state_num)}

	def _reset(self):
		self.env.reset()
		self.state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.traj = []
		self.total_reward = 0.0
		self.step_num = 0.0

	def take_action(self):
		self.epsilon = 0.9*0.99**self.epi + 0.1
		if random.uniform(0,1) < self.epsilon:
			action = np.random.choice(range(self.action_num))
		else: 
			action = np.argmax(self.q_table[self.state])

		_, reward, self.done, info = self.env.step(action)
		next_state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.traj.append(self.state)
		self.state_count[self.state] += 1
		self.total_reward += reward 
		self.step_num += 1
		return action, reward, next_state, self.done

	def update(self, action, reward, next_state):
		ones = np.zeros(self.state_num)
		ones[self.state] = 1
		m_target = ones + self.gamma*self.sr_matrix[next_state]
		m_error = m_target - self.sr_matrix[self.state]
		self.sr_matrix[self.state] += 0.1*m_error
		self.sf_matrix = np.dot(self.sr_matrix, self.state_f)
		v_target = reward + self.gamma*np.dot(self.sf_matrix[next_state], self.w)
		error = v_target - self.q_table[self.state, action]
		self.w += self.learning_rate*error*self.sf_matrix[self.state]
		self.q_table[self.state, action] = np.dot(self.sf_matrix[self.state], self.w)
		self.state = next_state


	def run(self):
		for epi in range(self.epi_num):
			self.epi = epi
			self._reset()
			done = False
			while not done:
				action, reward, next_state, done = self.take_action()
				self.update(action, reward, next_state)
				if self.step_num > self.max_step:
					break
			self.epi_total_reward.extend([self.total_reward])
			self.epi_step_num.extend([self.step_num])
			if epi % 1 == 0:
				print('SF ~~ epidose {}, total reward {}, step num {}'.format(epi, self.total_reward, self.step_num))
			if epi <= 20:
				self.all_traj[epi] = self.traj
		return self.epi_total_reward, np.array(self.state_count), self.all_traj











