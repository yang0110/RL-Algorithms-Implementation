import numpy as np 
import random 

class TD_tabular_prediction:
	'''
	Policy evaluation 
	'''
	def __init__(self, env, alpha, gamma, epsilon, epi_num):
		self.env = env 
		self.alpha = alpha
		self.gamma = gamma 
		self.epsilon = epsilon
		self.epi_num = epi_num
		self.v_table = np.zeros(self.env.observation_space.n)
		self.epi_total_reward = []
		self.epi_step_num = []
		self._reset()

	def _reset(self):

		self.state = self.env.reset()
		self.total_reward = 0.0 
		self.step_num = 0 

	def take_action(self):
		action = self.env.action_space.sample() # random policy 
		next_state, reward, done, info = self.env.step(action)

		return action, reward, next_state, done

	def update_value(self, action, reward, next_state):
		old_value = self.v_table[self.state]
		new_value = (1-self.alpha)*old_value + self.alpha*(reward + self.gamma*self.v_table[next_state])
		self.v_table[self.state] = new_value
		self.state = next_state

	def run(self): 
		for epi in range(self.epi_num):
			self._reset()
			done = False 
			while not done:
				action, reward, next_state, done = self.take_action()
				self.update_value(action, reward, next_state)
				self.total_reward += reward
				self.step_num += 1
			if epi % 100 == 0:
				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				print('TD ~~~ epi {}, total_reward {}, step_num {}'.format(epi, self.total_reward, self.step_num))
		return self.v_table, self.epi_total_reward, self.epi_step_num



class TD_tabular_control:
	'''
	Using V values for control, we need the transition matrix which could be given or estimated.
	'''


