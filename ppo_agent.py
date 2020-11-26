import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random 
from agent_utils import *
import collections 
import torch.nn.functional as F

Experience=collections.namedtuple('Experience', field_names=[
	'state','action', 'reward', 'done', 'next_state'])

class PPO:
	def __init__(self, env, epi_num, gamma, gae_gamma, learning_rate_actor, learning_rate_critic, traj_size, batch_size, ppo_eps, epoch_num):
		self.env = env 
		self.epi_num = epi_num 
		self.gamma = gamma 
		self.gae_gamma = gae_gamma
		self.learning_rate_actor = learning_rate_actor
		self.learning_rate_critic = learning_rate_critic
		self.traj_size = traj_size
		self.batch_size = batch_size
		self.ppo_eps = ppo_eps
		self.epoch_num = epoch_num
		self.epi_total_reward = []
		self.epi_step_num = []
		self.loss_list= []
		self._reset()
		self.init_net()

	def _init(self):
		self.state = self.env.reset()
		self.done = False 
		self.total_reward = 0.0
		self.step_num = 0
		self.trajectory = []

	def init_net(self):
		self.net_act = actor_net(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(device)
		self.net_crt = critic_net(self.env.observation_space.shape[0]).to(device)
		self.opt_act = optim.Adam(self.actor_net.parameters(), lr=self.learning_rate_actor)
		self.opt_crt = optim.Adam(self.critic_net.parameters(), lr=self.learning_rate_critic)

	def take_action(self):
		state_a = np.array([self.state], copy=False)
		state_v = torch.FloatTensor(state_a).to('cpu')
		logits_v, _ = self.net_act(state_v)
		probs_v = F.softmax(logits_v, dim=1)
		probs_a = probs_v.data.cpu().numpy().flatten()
		action = np.random.choice(len(probs_a), p = probs_a)

		next_state, reward, self.done, info = self.env.step(action)
		exp = Experience(self.state, action, reward, self.done, next_state)
		self.trajectory.append(exp)
		self.total_reward += reward
		self.step_num += 1 
		self.state = next_state

	def unpack_trajectory(self):
		self.traj_states = [exp.state for exp in self.trajectory]
		self.traj_actions = [exp.action for exp in self.trajectory]
		self.traj_states_v = torch.FloatTensor(traj_states).to('cpu')
		self.traj_actions_v = torch.FloatTensor(traj_actions).to('cpu')
		mu_v = self.net_crt(traj_states_v)
		old_logprob_v = cal_logprob(mu_v, net.act.logstd, traj_actions_v)
        # the logarithm of probability of the actions taken 
        traj_adv_v, self.traj_ref_v = cal_adv_ref(trajectory, net_crt, traj_states_v, device=device)

		self.traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v))/torch.std(traj_adv_v)
        # normalized advantage's mean and variance to improve the training stability.
        # trajectory = trajectory[:-1]
        self.old_logprob_v = old_logprob_v[:-1].detach()
        self.trajectory.clear()

	def update_net(self):
		for epoch in range(self.epoch_num):
			for batch_offset in range(0, len(self.traj_states), self.batch_size):
				states_v = self.traj_states_v[batch_offset:batch_offset+self.batch_size]
				actions_v = self.traj_actions_v[batch_offset:batch_offset+self.batch_size]
                batch_ref_v = self.traj_ref_v[batch_ofs:batch_ofs + ppo_batch_size]
                batch_old_logprob_v = self.old_logprob_v[batch_ofs:batch_ofs + ppo_batch_size]

                self.opt_crt.zero_grad()
                value_v = self.net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                loss_value_v.backward()
                self.opt_crt.step()

                # actor training
                # minimize the nagated clipped objective in PPO
                self.opt_act.zero_grad()
                mu_v = self.net_act(states_v)
                logprob_pi_v = calc_logprob(mu_v, self.net_act.logstd, actions_v)
                ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v * ratio_v
                clipped_surr_v = batch_adv_v * torch.clamp(ratio_v, 1.0 - ppo_eps, 1.0 + ppo_eps)
                loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                loss_policy_v.backward()
                opt_act.step()
                loss = loss_value_v.item()+loss_policy_v.item()
                self.loss_list.append(loss)


	def run(self):
		for epi in range(self.epi_num):
			self._reset()
			while not self.done:
				self.take_action()
				if len(self.trajectory) == self.traj_size:
					self.unpack_trajectory()
					self.update_net()

			self.epi_total_reward.append(self.total_reward)
			self.epi_step_num.append(self.step_num)
			if epi % 10 == 0:
				print('epi {}, total_reward {}, step_num {}, loss'.format(epi, self.epi_total_reward, self.step_num, self.loss_list[-1]))
		return self.epi_total_reward, self.epi_step_num, self.loss_list



