import gym 
import numpy as np 
import random
import matplotlib.pyplot as plt
import seaborn as sns
from agents import td_linear_agent
# sns.set_style("white")
result_path = 'results/'

env = gym.make('CartPole-v0') # continuous state space and discrete action space
alpha = 0.1 
gamma = 0.6
epsilon = 0.1 
epi_num = 10000

agent = td_linear_agent.TD_linear(env, alpha, gamma, epsilon, epi_num)
w, epi_td_error, epi_total_reward, epi_step_num = agent.run()

plt.figure(figsize=(5,5))
plt.plot(epi_total_reward)
plt.xlabel('episode index', fontsize=12)
plt.ylabel('total reward', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'cartpole_v0_td_linear_total_reward'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(epi_step_num)
plt.xlabel('episode index', fontsize=12)
plt.ylabel('step num', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'cartpole_v0_td_linear_step_num'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(epi_td_error)
plt.xlabel('episode index', fontsize=12)
plt.ylabel('td_error', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'cartpole_v0_td_linear_td_error'+'.png', dpi=100)
plt.show()
