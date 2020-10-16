import numpy as np 
import random 

class SARSA_tabular:
	'''
	SARSA is on-policy version of Q-Leaning
	'''
	def __init__(self, env, alpha, gamma, epsilon, epi_num):
	# '''
	# env: environment 
	# alpha: learning step
	# gamma: discount factor
	# epsilon: exploration rate 
	# epi_num: the number of episode (trajectory) to run until stop
	# '''
		self.env = env 
		self.alpha = alpha 
		self.gamma = gamma 
		self.epsilon = epsilon
		self.epi_num = epi_num
		self.epi_total_reward = []
		self.epi_step_num = []
		self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
		self._reset()

	def _reset(self):
		self.state = self.env.reset()
		self.total_reward = 0.0 
		self.step_num = 0.0

	def take_action(self):
		if random.uniform(0,1) < self.epsilon:
			action = self.env.action_space.sample()
		else: 
			action = np.argmax(self.q_table[self.state])
		next_state, reward, done, info = self.env.step(action)
		return action, reward, next_state, done

	def update_value(self, action, reward, next_state):
		old_value = self.q_table[self.state, action]
		if random.uniform(0,1) < self.epsilon/100.0:
			next_action = self.env.action_space.sample()
		else:
			next_action = np.argmax(self.q_table[next_state])

		new_value = (1-self.alpha)*old_value+self.alpha*(reward+self.gamma*self.q_table[next_state, next_action])
		self.q_table[self.state, action] = new_value
		self.state = next_state 

		self.total_reward += reward
		self.step_num += 1

	def run(self):
		for epi in range(self.epi_num):
			self._reset()
			done = False 

			while not done:
				action, reward, next_state, done = self.take_action()
				self.update_value(action, reward, next_state)

			if epi % 100 == 0:
				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				print('SARSA ~~~ epi {}, total reward {}, step num {}'.format(epi, self.total_reward, self.step_num))
		print('training finished! total reward {}, step num {}'.format(self.total_reward, self.step_num))
		return self.q_table, self.epi_total_reward, self.epi_step_num




















