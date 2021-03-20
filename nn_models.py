import numpy as np 
import collections
import torch
import torch.nn as nn
import math 
import torch.nn.functional as F

# class state2emb_ac_emb_net(nn.Module):


class AC_value_net(nn.Module):
    def __init__(self, state_num, dim, ac_matrix=None):
        super(AC_value_net, self).__init__()
        self.ac_matrix = ac_matrix
        # if self.ac_matrix is None:
        self.embedding = nn.Embedding(state_num, dim)
        # else:
            # self.embedding = nn.Embedding.from_pretrained(self.ac_matrix, freeze=True)

        self.linear1 = nn.Linear(dim, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1)

    def forward(self, states):
        emb = self.embedding(states).squeeze(1)
        x = self.linear1(emb)
        x = self.relu(x)
        states_value = self.linear2(x)
        return emb, states_value

class AC_policy_net(nn.Module):
    def __init__(self, action_num, dim):
        super(AC_policy_net, self).__init__()
        self.linear1 = nn.Linear(dim, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, action_num)
            
    def forward(self, emb):
        x = self.linear1(emb)
        x = self.relu(x)
        action_prob = self.linear2(x)
        return action_prob



class DSF_sf_nn(nn.Module):
    def __init__(self, state_num, dim, dsf_matrix=None):
        super(DSF_sf_nn, self).__init__()
        self.dsf_matrix = dsf_matrix
        if self.dsf_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(self.dsf_matrix, freeze=False)
            
        self.linear1 = nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, dim)
        self.relu = nn.ReLU()

    def forward(self, states):
        if self.dsf_matrix is None:
            state_embs = self.embedding(states).squeeze(1)
            x = self.linear1(state_embs)
            x = self.relu(x)
            state_sfs = self.linear2(x)
        else:
            state_sfs = self.embedding(states).squeeze(1)
            state_embs = state_sfs
        return state_embs, state_sfs

class DSF_q_nn(nn.Module):
    def __init__(self, action_num, dim):
        super(DSF_q_nn, self).__init__()
        self.linear = nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, action_num)
        self.relu = nn.ReLU()

    def forward(self, state_sfs):
        x = self.linear(state_sfs)
        x = self.relu(x)
        q_vals = self.linear2(x)
        return q_vals


class State2emb_embedding_nn(nn.Module):
    def __init__(self, state_num, action_num, dim, state2emb_matrix=None):
        super(State2emb_embedding_nn, self).__init__()
        if state2emb_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(state2emb_matrix, freeze=False)

    def forward(self, states):
        x = self.embedding(states).squeeze(1)
        # x = F.normalize(x, dim=0, p=2)
        cov = torch.mm(x, x.T)
        # matrix = torch.sigmoid(cov)
        return x, cov

class State2emb_q_nn(nn.Module):
    def __init__(self, action_num, dim):
        super(State2emb_q_nn, self).__init__()
        self.linear1= nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, action_num)
        self.relu = nn.ReLU()

    def forward(self, states_embedding):
        y = self.linear1(states_embedding)
        y = self.relu(y)
        q_values = self.linear2(y)
        # q_values = torch.sigmoid(q_values)
        return q_values

class DQN_emb_nn(nn.Module):
    def __init__(self, state_num, action_num, dim, dqn_matrix=None):
        super(DQN_emb_nn, self).__init__()
        # self.embedding = nn.Embedding(state_num, dim)
        if dqn_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(dqn_matrix, freeze=True)

    def forward(self, states):
        x = self.embedding(states).squeeze(1)
        return x

class DQN_q_nn(nn.Module):
    def __init__(self, action_num, dim):
        super(DQN_q_nn, self).__init__()
        self.linear1= nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, action_num)
        self.relu = nn.ReLU()
        
    def forward(self, states_embedding):
        y = self.linear1(states_embedding)
        y = self.relu(y)
        q_values = self.linear2(y)
        # q_values = torch.sigmoid(q_values)
        return q_values    



class DQN_nn(nn.Module):
    def __init__(self, state_num, action_num, dim, dqn_matrix=None):
        super(DQN_nn, self).__init__()
        # self.embedding = nn.Embedding(state_num, dim)
        if dqn_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(dqn_matrix, freeze=True)

        self.linear = nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, action_num)
        self.relu = nn.ReLU()

    def forward(self, states):
        x = self.embedding(states).squeeze(1)
        y = self.linear(x)
        y = self.relu(y)
        q_values = self.linear2(y)
        # q_values = torch.sigmoid(q_values)
        return q_values

class DQN_node2vec_nn(nn.Module):
    def __init__(self, state_num, action_num, dim, node2vec_matrix=None):
        super(DQN_node2vec_nn, self).__init__()
        if node2vec_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(node2vec_matrix, freeze=True)
        self.linear = nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, action_num)
        self.relu = nn.ReLU()

    def forward(self, states):
        x = self.embedding(states).squeeze(1)
        y = self.linear(x)
        y = self.relu(y)
        q_values = self.linear2(y)
        # q_values = torch.sigmoid(q_values)
        return q_values


class DQN_pvf_nn(nn.Module):
    def __init__(self, state_num, action_num, dim, pvf_matrix=None):
        super(DQN_pvf_nn, self).__init__()
        if pvf_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(pvf_matrix, freeze=True)

        self.linear = nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, action_num)
        self.relu = nn.ReLU()

    def forward(self, states):
        x = self.embedding(states).squeeze(1)
        y = self.linear(x)
        y = self.relu(y)
        q_values = self.linear2(y)
        # q_values = torch.sigmoid(q_values)
        return q_values



class A2C_net(nn.Module):
    def __init__(self, input_size, n_actions):
        super(A2C_net, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
            )

        self.value_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )

    def forward(self,x):
        action_prob = self.policy_net(x)
        state_value = self.value_net(x)
        return action_prob, state_value



class A2C_net_ca(nn.Module):
    '''
    Actot-critic net for continuous action (ca) space .

    Parameter
    ---------
    Input_size: observation dimension
    action_size: action feature diemension
    mu: return the mean of output action
    var: the variance of output action
    value: the state value
    '''
    def __init__(self, input_size, action_size):
        super(A2C_net_ca, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            )
        self.mu = nn.Sequential(
            nn.Linear(32, action_size),
            nn.Tanh(),
            )
        self.var = nn.Sequential(
            nn.Linear(32, action_size),
            nn.Softplus(),
            )
        self.value = nn.Linear(32, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)



class DDPG_actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ddpg_actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, act_size), 
            nn.Tanh()
            )
    def forward(self, x):
        return self.net(x)

class DDPG_critic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ddpg_critic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            )
        self.out_net = nn.Sequential(
            nn.Linear(64+act_size, 32),
            nn.ReLU(),
            nn.Linear(32,1)
            )
    def forward(self, x,a):
        obs = self.obs_net(x)
        out = self.out_net(torch.cat([obs, a], dim=1))
        return out 






class Actor(nn.Module):
    '''
    The network is designed for continuous action

    parameters:
    ----------
    obs_size: observation dimension
    act_size: action vetor dimension

    return: 
    ------
    mean of action vector
    '''
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(), 
            np.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_size),
            nn.Tanh(),
            )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)

class Critic(nn.Module):
    '''
    calculate the state value

    parameter:
    ---------
    obs_size: observation dimension

    return:
    ------
    state value (scaler)
    '''
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(obs_size, 64), 
            nn.ReLU(), 
            nn.Linear(64,64), 
            nn.ReLU(),
            nn.Linear(64, 1),
            )
    def forward(self, x):
        return self.value(x)









