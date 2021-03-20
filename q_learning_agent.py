import numpy as np 
import random 

class Q_learning:
	def __init__(self, env, size, state_num, beta, alpha, gamma, epsilon, epi_num, max_step):
		self.env = env 
		self.alpha = alpha 
		self.beta = beta
		self.gamma = gamma 
		self.epsilon = epsilon
		self.size= size
		self.state_num = state_num 
		self.action_num = self.env.action_space.n
		self.q_table = np.zeros((self.state_num, self.action_num))
		self.ir_table = np.zeros((self.state_num, self.action_num))
		self.epi_num = epi_num 
		self.max_step = max_step
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi= 0
		self.traj = []
		self.all_traj = {}
		self.state_neighbors = {i: [] for i in range(self.state_num)}
		self.state_count = [0 for i in range(self.state_num)]
		self._reset() 

	def _reset(self):
		self.env.reset()
		self.state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.state_neighbors = {i: [] for i in range(self.state_num)}
		self.total_reward = 0.0 
		self.step_num = 0.0
		self.traj = []
		self.done = False

	def take_action(self):
		self.epsilon = 0.9*0.99**self.epi + 0.1
		if random.uniform(0,1) < self.epsilon:
			action = np.random.choice(range(self.action_num))
		else: 
			action = np.argmax(self.q_table[self.state])

		_, reward, self.done, info = self.env.step(action)
		next_state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.state_neighbors[self.state].append(next_state)
		self.state_neighbors[self.state] = np.unique(self.state_neighbors[self.state]).tolist()
		self.state_count[self.state] += 1
		self.total_reward += reward 
		self.step_num += 1
		# ir = self.beta*(1/(np.sqrt(self.state_count[next_state]+1)*np.sqrt(len(self.state_neighbors[next_state])+1)))
		# self.ir_table[self.state, action] = ir
		# reward += ir
		self.traj.append(next_state)
		return action, reward, next_state, self.done

	def update_q_table(self, action, reward, next_state):
		old_value = self.q_table[self.state, action]
		new_value = self.alpha*(reward+self.gamma*np.max(self.q_table[next_state]))+(1-self.alpha)*old_value
		self.q_table[self.state,action] = new_value
		self.state = next_state

	def run(self):
		for epi in range(self.epi_num):
			self.epi = epi
			self._reset()
			done = False
			while not done:
				action, reward, next_state, done = self.take_action()
				self.update_q_table(action, reward, next_state)
				if self.step_num > self.max_step:
					break
			if epi % 1 == 0:
				print('Q_learning ~~ epidose {}, total reward {}, step num {}'.format(epi, np.round(self.total_reward,decimals=2), self.step_num))

			if (epi<=10) or (epi>=self.epi_num-10):
				self.all_traj[epi] = self.traj

			self.epi_total_reward.extend([self.total_reward])
			self.epi_step_num.extend([self.step_num])
		return self.epi_total_reward, self.state_num











