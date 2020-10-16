import numpy as np 
import random 

class TD_linear:
	'''
	Policy prediction
	'''
	def __init__(self, env, alpha, gamma, epsilon, epi_num):
		self.env = env 
		self.alpha = alpha 
		self.gamma = gamma 
		self.epsilon = epsilon
		self.epi_num = epi_num
		self.dimension = self.env.observation_space.shape[0]
		self.w = np.zeros(self.dimension)
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi_td_error = []
		self._reset()

	def _reset(self):
		self.state = self.env.reset()
		self.total_reward = 0.0 
		self.step_num = 0 

	def take_action(self):
		action = self.env.action_space.sample()
		next_state, reward, done, info = self.env.step(action)
		return action, reward, next_state, done 

	def update_parameter(self, action, reward, next_state):
		target = reward + self.gamma*np.dot(next_state, self.w)
		self.td_error = target - np.dot(self.state, self.w)
		self.w += self.alpha*self.td_error*self.state
		self.state = next_state

	def run(self):
		for epi in range(self.epi_num):
			self._reset()
			done = False 
			while not done:
				action, reward, next_state, done = self.take_action()
				self.update_parameter(action, reward, next_state)
				self.total_reward += reward
				self.step_num += 1 

			if epi % 100 == 0:
				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				self.epi_td_error.extend([self.td_error])
				print('TD-Linear ~~~ epi {}, total reward {}, step num {}'.format(epi, self.total_reward, self.step_num))

		print('TD_linear ~~~ Training Finished!')
		return self.w, self.epi_td_error, self.epi_total_reward, self.epi_step_num







