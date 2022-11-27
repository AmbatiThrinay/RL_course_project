# Author : Ambati Thrinay Kumar Reddy
#  Deep Q Neural network Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, n_inputs, hid1_dims, hid2_dims, n_actions, seed):
        '''
        Deep Neural Network model

        params
        ======
            n_inputs (int): Dimensions of each obs
            hid1_dims (int): Number of nodes in first hidden layer
            hid2_dims (int): Number of nodes in second hidden layer
            n_actions (int): Number of actions
            seed (int) : Random seed for random number generator
        '''
        super(DeepQNetwork, self).__init__() # intialize the nn.Module class

        # generator the same intial weights for the network
        self.seed = torch.manual_seed(seed)

        # Network layers
        self.fc1 = nn.Linear(n_inputs, hid1_dims)
        self.fc2 = nn.Linear(hid1_dims, hid2_dims)
        self.fc3 = nn.Linear(hid2_dims, n_actions)

    def forward(self, obs):
        '''
        Build the neural network that maps states(x) to action probabilities
        '''
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save(self, filepath):
        print(f'..saving checkpoint .. at {filepath}')
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        print(f'... loading checkpoint ... from {filepath}')
        self.load_state_dict(torch.load(filepath))

class DQN():
    def __init__(
        self, n_obs:int, n_actions:int, nn_lr:float, discount:float, batch_size:int,
        seed:int, eps_max:float, eps_min:float=0.01, eps_decay_rate:float=5e-4, memory_size:int=100_000,
        target_update_rate:int = 100):
        '''
        ### Description
        Deep Q Netwerk agent

        <-- Input args -->
        n_obs (int): Dimensions of each obs
        n_actions (int): Number of actions
        ** see DQN config **
        '''
        self.gamma = discount
        self.epsilon = eps_max
        self.eps_min = eps_min
        self.eps_dec = eps_decay_rate
        self.lr = nn_lr
        self.action_space = list(range(n_actions))
        self.mem_size = memory_size
        self.batch_size = batch_size
        self.mem_cntr = 0 # memory counter - no of experiences considered
        self.learn_step_cntr = 0 # for updating the target network for every few steps
        self.replace_rate = target_update_rate

        # train on GPU if available
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

        self.Q_network_local = DeepQNetwork(n_obs,32, 64, n_actions, seed).to(self.device)
        self.Q_network_target = DeepQNetwork(n_obs,32, 64, n_actions, seed).to(self.device)

        self.optimizer = torch.optim.Adam(self.Q_network_local.parameters(), lr=nn_lr) # Adam optimizer
        self.loss = nn.HuberLoss() # Huber loss

        # Replay memory
        self.state_memory = np.zeros((self.mem_size, n_obs), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, n_obs), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)

    def store_transition(self, obs, action, reward, next_state, done):
        '''
        Add new experience in the buffer
        '''
        index = self.mem_cntr % self.mem_size # index of memory to be overwritten
        self.mem_cntr += 1

        self.state_memory[index] = obs
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

    def choose_action(self, obs):

        # epsilon greedy policy
        if np.random.random() > self.epsilon:
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            actions = self.Q_network_local.forward(obs)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
        
    def learn(self):

        # learning starts only after the memory has batch size of experience
        if self.mem_cntr < self.batch_size: return

        # Update the target network for every 'repace_rate' steps
        if self.learn_step_cntr % self.replace_rate == 0:
            self.Q_network_target.load_state_dict(self.Q_network_local.state_dict())
        
        # randomly select only the batch size experiences(not zeroes)
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem,self.batch_size,replace=False)

        batch_index = np.arange(self.batch_size,dtype=np.int32) 
        state_batch = torch.tensor(self.state_memory[batch]).to(self.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.device)   
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.device)
        action_batch = self.action_memory[batch]

        # Q predict from local network and Q target from target network for TD error
        q_pred = self.Q_network_local.forward(state_batch)[batch_index,action_batch]
        q_next =  self.Q_network_local.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next,dim=1)[0]

        # training the local neural network
        self.optimizer.zero_grad()
        loss = self.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.optimizer.step()
        self.learn_step_cntr += 1

    # annealing the exploratory rate(epsilon)
    def anneal_epsilon(self):
        if self.epsilon > self.eps_min: self.epsilon -= self.eps_dec
        else: self.epsilon = self.eps_min
    
    def save(self, filepath):
        self.Q_network_local.save(filepath+'_local.pth')
        self.Q_network_target.save(filepath+'_target.pth')

    def load(self, filepath):
        self.Q_network_local.load(filepath)
        self.Q_network_target.load(filepath)
