import gym 
import numpy as np 
import random
import matplotlib.pyplot as plt
import seaborn as sns
from agents import q_learning_tabular_agent, sarsa_tabular_agent, td_tabular_agent
# sns.set_style("white")
result_path = 'results/'

env = gym.make('Taxi-v3')
alpha = 0.1 
gamma = 0.6
epsilon = 0.1 
epi_num = 10000

agent = q_learning_tabular_agent.Q_learning_tabular(env, alpha, gamma, epsilon, epi_num)
q_table, epi_total_reward, epi_step_num = agent.run()

agent2 = sarsa_tabular_agent.SARSA_tabular(env, alpha, gamma, epsilon, epi_num)
q_table2, epi_total_reward2, epi_step_num2 = agent2.run()

agent3 = td_tabular_agent.TD_tabular(env, alpha, gamma, epsilon, epi_num)
v_table, epi_total_reward3, epi_step_num3 = agent3.run()

# q_table = np.zeros((env.observation_space.n, env.action_space.n))

# all_epochs = []
# all_penalties = []

# epi_total_reward = []
# epi_step_num = []

# for epi_index in range(epi_num):
# 	state = env.reset()
# 	epochs, penalties, reward = 0, 0, 0
# 	done = False 
# 	total_reward = 0.0 
# 	step_num = 0

# 	while not done:
# 		if random.uniform(0,1) < epsilon:
# 			action = env.action_space.sample()
# 		else:
# 			action = np.argmax(q_table[state])

# 		next_state, reward, done, info = env.step(action)
# 		old_value = q_table[state, action]
# 		next_max = np.max(q_table[next_state])
# 		new_value = (1-alpha)*old_value+alpha*(reward+next_max)
# 		q_table[state, action] = new_value
# 		state = next_state 

# 		total_reward += reward 
# 		step_num += 1

# 	if epi_index % 100 == 0:
# 		print('episode {}, total reward {}, step num {}'.format(epi_index, total_reward, step_num))
# 		epi_total_reward.extend([total_reward])
# 		epi_step_num.extend([step_num])

# print('Training finished with episode {}, reward {}, steps {}'.format(epi_num, total_reward, step_num))

plt.figure(figsize=(5,5))
plt.plot(epi_total_reward)
plt.xlabel('episode index', fontsize=12)
plt.ylabel('total reward', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'taxi_v3_q_learning_tabular_total_reward'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(epi_step_num)
plt.xlabel('episode index', fontsize=12)
plt.ylabel('step num', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'taxi_v3_q_learning_tabular_step_num'+'.png', dpi=100)
plt.show()



plt.figure(figsize=(5,5))
plt.plot(epi_total_reward2)
plt.xlabel('episode index', fontsize=12)
plt.ylabel('total reward', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'taxi_v3_sarsa_tabular_total_reward'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(epi_step_num2)
plt.xlabel('episode index', fontsize=12)
plt.ylabel('step num', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'taxi_v3_sarsa_tabular_step_num'+'.png', dpi=100)
plt.show()
