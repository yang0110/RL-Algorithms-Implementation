import numpy as np 
import random 

class LSPI:
	'''
	LSPI is designed based on LSTD with the difference that Q-value is approximated instead of V-values. 
	With Q-values, policy can be improved by acting greedy. 
	Another algorithm named LSTD_Q is the policy evaluation part of LSPI.
	'''
	def __init__(self, env, alpha, gamma, epsilon, epi_num):
		self.env = env 
		self.alpha = alpha 
		self.gamma = gamma 
		self.epsilon = epsilon
		self.epi_num = epi_num
		self.dimension = self.env.observation_space.shape[0]+action_dimension
		self.w = np.zeros(self.dimension)
		self.A = np.identity(self.dimension)*0.1 
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

	def feature_encoder(self, state, action):
		'''
		In gym env, the observation is treated as the state feature. However, we need state-action feature. 
		Thus, a feature encoder is needed to generate state-action feature based on state and action.
		'''

	def action_feature(self, action):
		self.state_action_f = feature_encode(self.state, action)

	def take_action(self):
		if random.uniform(0,1) < self.epsilon:
			action = self.env.action_space.sample()
		else:
			action_num = self.env.action_space.n 
			state_action_f_m = np.zeros((action_num, self.dimension))
			for i in range(action_num):
				state_action_f_m[i] = feature_encode(self.state, i)
			q_values = np.dot(state_action_f_m, self.w)
			action = np.argmax(q_values)

		next_state, reward, done, info = self.env.step(action)
		self.td_error = np.dot(self.state, self.w)-(reward+self.gamma*np.dot(next_state, self.w))
		return action, reward, next_state, done

	def update_parameter(self, action, reward, next_state):
		x = feature_encode(self.state, action)
		x_p = feature_encode(next_state, next_action)
		self.A += np.outer(x, x-self.gamma*x_p)
		self.b += np.dot(x, reward)
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
				print('LSPI ~~~ epi {}, total reward {}, step num {}'.format(epi, total_reward, step_num))

		print('LSPI Training Finished!')
		return self.w, self.epi_td_error, self.epi_total_reward, self.epi_step_num

