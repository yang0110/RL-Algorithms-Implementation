import numpy as np 
import random 

class LSTD:
	'''
	policy prediction
	'''
	def __init__(self, env, alpha, gamma, epsilon, epi_num):
		self.env = env 
		self.alpha = alpha 
		self.gamma = gamma 
		self.epsilon = epsilon
		self.epi_num = epi_num
		self.dimension = self.env.observation_space.shape[0]
		self.w = np.zeros(self.dimension)
		self.A = np.zeros((self.dimension, self.dimension))
		self.b = np.zeros(self.dimension)
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi_td_error = []
		self._reset()

	def _reset(self):
		self.state = self.env.reset()
		self.total_reward = 0.0 
		self.step_num = 0 
		self.td_error = 0.0 

	def take_action(self):
		action = self.env.action_space.sample() # random policy
		next_state, reward, done, info = self.env.step(action)
		self.td_error = np.dot(self.state, self.w)-(reward+self.gamma*np.dot(next_state, self.w))
		return action, reward, next_state, done

	def update_parameter(self, action, reward, next_state):
		self.A += np.outer(self.state, self.state-self.gamma*next_state)
		self.b += np.dot(self.state, reward)
		self.w = np.dot(np.linalg.pinv(self.A), self.b)
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
				print('LSTD ~~~ epi {}, total reward {}, step num {}'.format(epi, total_reward, step_num))

		print('LSTD Training Finished!')
		return self.w, self.epi_td_error, self.epi_total_reward, self.epi_step_num








