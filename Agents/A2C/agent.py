from collections import deque, namedtuple
import random
import torch.nn as nn
import torch 
import torch.optim as optim


L_R = 3e-4
GAMMA = 0.99
ENTROPY_BETA = 0.01
BATCH_SIZE = 64


MEMORY_CAPACITY = 30_000

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class Memory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        """Add a new transition to memory."""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        """Sample a batch of transitions from memory."""
        if len(self.memory) < batch_size:
            return []
        batch = random.sample(self.memory, batch_size)
        return batch
    
    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        """Clear the memory."""
        self.memory.clear()

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size=3):
        super(ActorNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_size, 400),
            nn.LayerNorm(400),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    


class Agent:
    def __init__(self, state_size, action_size,device , lr=L_R, gamma=GAMMA, entropy_beta=ENTROPY_BETA, batch_size=BATCH_SIZE, memory_capacity=MEMORY_CAPACITY):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.device = device
        
        self.actor = ActorNetwork(state_size, action_size).to(self.device)
        self.critic = CriticNetwork(state_size).to(self.device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.memory = Memory(self.memory_capacity)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return  # Return or handle appropriately if there aren't enough samples
    
        transitions = self.memory.sample(self.batch_size)
        if not transitions:
            return
        
        batch = Transition(*zip(*transitions))
        
        states = torch.cat([torch.FloatTensor(state).unsqueeze(0).to(self.device) for state in batch.state])
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = torch.cat([torch.FloatTensor(state).unsqueeze(0).to(self.device) for state in batch.next_state])
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        # Compute values and next values
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()

        # Compute returns
        returns = rewards + self.gamma * next_values * (1 - dones)

        # Compute advantages
        advantages = returns - values

        # Compute the new policy
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Compute actor loss
        actor_loss = -(new_log_probs * advantages.detach()).mean() - self.entropy_beta * entropy

        # Compute critic loss
        critic_loss = nn.MSELoss()(values, returns.detach())

        # Optimize
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.memory.clear()

    def reset(self):
        self.memory.clear()