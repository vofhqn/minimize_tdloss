import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from numpy.random import choice
from torch.distributions import Normal
mpl.use('Agg') 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 1e-4

data = np.load("Hopper_data.npz")
numdata = data['state'].shape[0]
state_dim=data['state'].shape[1]
action_dim=data['action'].shape[1]
train_steps = 2000
num_layers = 1
epsilon = 1e-6


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, num_layers, hidden_dim=200):

        super(QNet, self).__init__()

        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(state_dim + action_dim, 1)
        else:
            self.linears = torch.nn.ModuleList()        
            self.linears.append(nn.Linear(state_dim + action_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, 1))


    def forward(self, state, action):

        x = torch.cat([state, action], 1)
        if self.num_layers == 1:
            return self.linear(x)
        for layer in range(self.num_layers - 1):
            x = F.relu(self.linears[layer](x))
        return self.linears[self.num_layers - 1](x)

class ReplayMemory:
    def __init__(self, capacity, seed, state_dim, action_dim):
        random.seed(seed)
        self.capacity = capacity
        self.position = 0
        self.num_samples = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros(capacity)
        self.nextstates = np.zeros((capacity, state_dim))
        self.nextactions = np.zeros((capacity, action_dim))
        self.masks = np.zeros(capacity)
        self.weights = np.ones(capacity)

    def sample(self, batch_size):
        sum_w = np.sum(self.weights[:self.num_samples])
        batch = list(choice(self.num_samples, batch_size, p=self.weights[:self.num_samples] / sum_w))
        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        nextstates = self.nextstates[batch]
        masks = self.masks[batch]
        return states, actions, rewards, nextstates, masks

    def __len__(self):
        return self.num_samples

memory = ReplayMemory(1000000, 0, state_dim, action_dim)
memory.num_samples = numdata
memory.states = data['state']
memory.actions = data['action']
memory.rewards = data['reward']
memory.nextstates = data['nextstate']
memory.masks = data['mask']



LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=200, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)


        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        #std = torch.ones(std.shape).to(device) * 0.1
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


checkpoint=torch.load("policy_params")
policy = GaussianPolicy(state_dim, action_dim).to(device=device)
policy.load_state_dict(checkpoint['policy'])

def td_loss(memory, critic, critic_target, batch_size=256, gamma=0.99):
    states, actions, rewards, next_states, masks = memory.sample(batch_size)
    states = torch.FloatTensor(states).to(device)
    actions = torch.FloatTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    masks = torch.FloatTensor(masks).reshape(-1, 1).to(device)

    with torch.no_grad():
        _, n_a_log_prob, n_action = policy.sample(next_states)#expectation is used
        n_qf = critic_target(next_states, n_action)
        n_qvalue = rewards + gamma * masks * n_qf

    qf = critic(states, actions)
    qf_loss = F.mse_loss(qf, n_qvalue)
    return qf_loss


#Minimizing TDloss with SGD
Q = QNet(state_dim, action_dim, num_layers).to(device)
sgdloss = []
optimizer = torch.optim.Adam(Q.parameters(), lr=lr)
for t in range(train_steps):
    optimizer.zero_grad()
    loss = td_loss(memory, Q, Q)
    loss.backward()
    optimizer.step()
    sgdloss.append(float(loss.cpu().data.numpy()))

#minimizing TDloss with soft target update
Q = QNet(state_dim, action_dim, num_layers).to(device)
targetQ = copy.deepcopy(Q)
softloss = []
s_trueloss = []
optimizer = torch.optim.Adam(Q.parameters(), lr=lr)
for t in range(train_steps):
    optimizer.zero_grad()
    loss = td_loss(memory, Q, targetQ)
    loss.backward()
    optimizer.step()
    
    
    trueloss = td_loss(memory, Q, Q)
    s_trueloss.append(float(trueloss.cpu().data.numpy()))
    softloss.append(float(loss.cpu().data.numpy()))
    soft_update(targetQ, Q, 0.005)

line_sgd, =plt.plot(list(range(len(sgdloss))), np.log(sgdloss),  color='red', label="SGD")
line_soft, =plt.plot(list(range(len(softloss))), np.log(softloss), '--', color='yellow', label="soft")
line_soft, =plt.plot(list(range(len(softloss))), np.log(s_trueloss),  color='black', label="soft true")
plt.ylabel("log 10")
plt.legend()

plt.savefig("linear.jpg")