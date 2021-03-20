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

class AC:
	def __init__(self, env, size, state_num, gamma, dim, epi_num, max_step, learning_rate, entropy_beta, ac_matrix=None):
		self.env = env 
		self.size = size 
		self.state_num = state_num
		self.action_num = self.env.action_space.n
		self.dim = dim
		self.ac_matrix = ac_matrix
		self.gamma = gamma 
		self.epi_num = epi_num 
		self.max_step = max_step
		self.learning_rate = learning_rate
		self.entropy_beta = entropy_beta
		self.epi_total_reward = []
		self.epi_step_num = []
		self.epi_loss = []
		self.state_count = np.zeros(self.state_num)
		self.init_net()
		self._reset()

	def _reset(self):
		self.env.reset()
		self.state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		self.done = False
		self.total_reward = 0.0 
		self.step_num = 0
		self.episode_exp = []

	def init_net(self):
		self.value_net = AC_value_net(self.state_num, self.dim, self.ac_matrix)
		self.policy_net = AC_policy_net(self.action_num, self.dim)
		self.value_optimizer = optim.Adam(self.value_net.parameters(), lr = self.learning_rate)
		self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr = self.learning_rate)

	def take_action(self):
		state_a = np.array([self.state], copy=False)
		state_v = torch.LongTensor(state_a).to(device)
		state_emb, _ = self.value_net(state_v)
		logit_v = self.policy_net(state_emb)
		probs_v = F.softmax(logit_v, dim=1)
		probs_a = probs_v.data.cpu().numpy().flatten()
		action = np.random.choice(len(probs_a), p = probs_a)

		_, reward, self.done, info = self.env.step(action)
		next_state = self.size*self.size*self.env.agent_dir + (self.env.agent_pos[1]-1)*self.size+(self.env.agent_pos[0]-1)
		exp = Experience(self.state, action, reward, self.done, next_state)
		self.episode_exp.append(exp)
		self.total_reward += reward
		self.step_num += 1 
		self.state_count[self.state] += 1
		self.state = next_state

	def update_net(self):
		states = []
		next_states = []
		actions = []
		rewards = []
		for step, exp in enumerate(self.episode_exp):
			states.append(exp.state)
			next_states.append(exp.next_state)
			actions.append(int(exp.action))
			rewards.append(exp.reward)

		states_v = torch.LongTensor(states)
		next_states_v = torch.LongTensor(next_states)
		actions_v = torch.LongTensor(actions)
		rewards_v = torch.FloatTensor(rewards)

		states_emb_v, states_val_v = self.value_net(states_v)
		_, next_states_val_v = self.value_net(next_states_v)
		ref_vals_v = torch.reshape(rewards_v, (len(rewards_v),1)) + self.gamma*next_states_val_v
		self.value_optimizer.zero_grad()
		loss_val_v = nn.MSELoss()(states_val_v, ref_vals_v)
		loss_val_v.backward()
		self.value_optimizer.step()

		logits_v = self.policy_net(states_emb_v.detach())
		log_prob_v = F.log_softmax(logits_v, dim=1)
		adv_v = ref_vals_v.detach() - states_val_v.detach()
		log_prob_actions_v = adv_v*log_prob_v[range(len(states_v)), actions_v]
		loss_policy_v = -log_prob_actions_v.mean()
		prov_v = F.softmax(logits_v, dim=1)
		loss_entropy_v = self.entropy_beta*(prov_v*log_prob_v).sum(dim=1).mean()
		policy_loss = loss_policy_v + loss_entropy_v
		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		self.policy_optimizer.step()

		loss = loss_val_v + policy_loss
		return loss.item()

	def run(self):
		for epi in range(self.epi_num):
			self._reset()
			while not self.done:
				self.take_action()
				if self.step_num > self.max_step:
					break
			loss = self.update_net() # update net at the end of each episode.
			if epi % 1 == 0:
				self.epi_total_reward.extend([self.total_reward])
				self.epi_step_num.extend([self.step_num])
				self.epi_loss.extend([loss])
				print('AC ~~~ epi {}, total_reward {}, step num {}'.format(epi, self.total_reward, self.step_num))
		print('AC ~~~ Training Finished !')
		embedding = self.value_net.embedding.weight.cpu().detach().numpy()
		return self.epi_total_reward, embedding, self.state_count			


