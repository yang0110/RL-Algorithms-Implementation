import numpy as np 
import random 

class Q_learning:
	'''
	Q-Learning Agent for tabular RL problem.
	'''
	def __init__(self, env, alpha, gamma, epsilon, epi_num):

	# '''
	# 	env: environemnt
	# 	alpha: learning step 
	# 	gamma: discount factor
	# 	epsilon: exploration rate
	# 	epi_num: the number of episode (trajectory) to run before stop
	# 	epi_total_reward: a list to record the total reward of each episode
	# 	epi_step_num: a list records the number of step until done of each episode

	# '''
		self.env = env 
		self.alpha = alpha 
		self.gamma = gamma 
		self.epsilon = epsilon
		self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
		self.epi_num = epi_num 
		self.epi_total_reward = []
		self.epi_step_num = []
		self._reset() 

	def _reset(self):
		'''
		Reset the agent and environment at the end of each episode
		'''
		self.state = self.env.reset()
		self.total_reward = 0.0 
		self.step_num = 0.0

	def take_action(self):
		'''
		epsilon-greedy exploration
		'''
		if random.uniform(0,1) < self.epsilon:
			action = self.env.action_space.sample()
		else: 
			action = np.argmax(self.q_table[self.state])

		next_state, reward, done, info = self.env.step(action)
		self.total_reward += reward 
		self.step_num += 1
		return action, reward, next_state, done

	def update_q_table(self, action, reward, next_state):
		'''
		Update q value
		'''
		old_value = self.q_table[self.state, action]
		new_value = self.alpha*(reward+self.gamma*np.max(self.q_table[next_state]))+(1-self.alpha)*old_value
		self.q_table[self.state,action] = new_value
		self.state = next_state

	def run(self):
		for epi in range(self.epi_num):
			self._reset()
			done = False
			while not done:
				action, reward, next_state, done = self.take_action()
				self.update_q_table(action, reward, next_state)
			if epi % 100 ==0:
				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				print('Q_learning ~~ epidose {}, total reward {}, step num {}'.format(epi, self.total_reward, self.step_num))
		return self.q_table, self.epi_total_reward, self.epi_step_num











