from collections import deque, namedtuple
import random
import torch.nn as nn
import torch 
import math
import torch.optim as optim



L_R = 3e-4
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 40000
BATCH_SIZE = 64
MEMORY_CAPACITY = 30_000
GAMMA = 0.4


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class Memory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return []
        batch = random.sample(self.memory, batch_size)
        return batch 

    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        self.memory.clear()

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm (256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.LayerNorm (256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, action_size)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
class Agent:
    def __init__(self, state_size, action_size, device, lr=L_R, batch_size=BATCH_SIZE, memory_capacity=MEMORY_CAPACITY, epsilon = EPSILON, epsilon_min= EPSILON_MIN, epsilon_decay = EPSILON_DECAY, gamma = GAMMA):
        self.state_size = state_size # number of previous days * number of features
        self.action_size = action_size # buy, sell, hold
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.device = device 
        self.lr = lr
        self.memory = Memory(self.memory_capacity)

        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        
        self.target_net.eval() # fix the target net
        self.target_net.load_state_dict(self.policy_net.state_dict())

        

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
         


    def act(self, state):
        self.steps_done += 1
        
        # Calculate the epsilon threshold
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() > eps_threshold:
            with torch.no_grad():
                action_values = self.policy_net(state)
            action = torch.argmax(action_values).item()  # Extracting the index of the maximum value as a scalar
            
        else:
            action = random.choice(range(self.action_size))  # Randomly choosing an action index
            
        return action


    def optimize(self):
        if len(self.memory) < self.batch_size:
            return  # Return or handle appropriately if there aren't enough samples

        # Sample a batch of experiences from the replay memory 
        transitions = self.memory.sample(self.batch_size)
        if not transitions:
            return

        batch = Transition(*zip(*transitions))
        # print(f"This is states tensor: {batch.state}")
        # print(f"This is actions tensor: {batch.action}, THIS IS SIZE: {len(batch.action)}")
        # print(f"This is rewards tensor: {batch.reward}")
        # print(f"This is next_states tensor: {batch.next_state}")
        # print(f"This is dones tensor: {batch.done}")
        
        # Convert batch to tensors 
        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(batch.action).to(self.device).unsqueeze(1)
        rewards = torch.cat(batch.reward).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        # print(f"actions  {actions}, THIS IS SIZE: {len(batch.action)}")
        # print(f"rewards  {rewards}")
        # print(f"next_states  {next_states}")
        # print(f"dones  {dones}")

        current_q_values = self.policy_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        self.target_net.load_state_dict(self.policy_net.state_dict())
           
    def reset(self):
        self.memory.clear()