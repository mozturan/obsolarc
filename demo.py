# A SAC implementation using PyTorch for a simple continuous control task

import gc
import torch # PyTorch library
import torch.nn as nn # Neural network module
import torch.optim as optim # Optimization algorithms
import gymnasium as gym # Gymnasium for environments
import numpy as np # NumPy for numerical operations
from buffer import ReplayBuffer # Import ReplayBuffer from buffer.py

# Define the Actor network
class Actor(nn.Module):

    # Initialize the Actor network
    def __init__(self, state_dim, action_dim, hidden_dim,max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    # Define the forward pass
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)  #! Limit log_std to avoid numerical issues
        std = torch.exp(log_std)    
        return mu, std # Return mean and standard deviation of action distribution
    
    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z) * self.max_action # Scale action to the environment's action range
        
        #! Look for correct log prob calculation 
        log_prob = normal.log_prob(z)
        log_prob_minus= torch.log(1 - action.pow(2) + 1e-6) # Correction for Tanh squashing
        log_prob = log_prob - log_prob_minus
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob # Return sampled action and log probability
    
# Define the Critic network
class Critic(nn.Module):

    # Initialize the Critic network
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)

    # Define the forward pass
    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        q_value = self.q_head(x)
        return q_value
    
# Soft Actor-Critic (SAC) Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, 
                 buffer_size=int(1e6), min_buffer_size=1000,
                 batch_size=256, hidden_dim=256, max_action=1.0,
                 actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size, state_dim=state_dim, action_dim=action_dim)
        self.min_buffer_size = min_buffer_size # Minimum buffer size before training
        self.batch_size = batch_size # Batch size for training

        # Initialize Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, 
                           hidden_dim, max_action).to(self.device) # Initialize Actor network
        self.critic1 = Critic(state_dim, action_dim, 
                             hidden_dim).to(self.device) # Initialize Critic network
        self.critic2 = Critic(state_dim, action_dim, 
                             hidden_dim).to(self.device) # Initialize Critic network
        self.critic_target1 = Critic(state_dim, action_dim, 
                                    hidden_dim).to(self.device) # Initialize target Critic network
        self.critic_target2 = Critic(state_dim, action_dim, 
                                    hidden_dim).to(self.device) # Initialize target Critic network
        
        # Copy weights from Critic to target Critic networks
        self.critic_target1.load_state_dict(self.critic1.state_dict()) # Copy weights to target
        self.critic_target2.load_state_dict(self.critic2.state_dict()) # Copy weights to target

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr) # Actor optimizer
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr) # Critic optimizer
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr) # Critic optimizer
        
        self.gamma = gamma # Discount factor
        self.tau = tau # Soft update factor
        self.alpha = alpha # Entropy coefficient

    # Select action based on current policy
    def choose_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        
        if evaluate:

            #! This can be optimized further: refactor to avoid code duplication
            mu, _ = self.actor.forward(state)
            action = torch.tanh(mu) * self.actor.max_action
            return action.cpu().data.numpy() # Return action for evaluation
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().data.numpy()
        
    # Soft update target networks
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # Update the SAC agent
    def train(self, replay_buffer, batch_size=256):
        
        if self.replay_buffer.size < self.min_buffer_size:
            return  # Not enough data to train
        
        # Sample a batch of transitions from the replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Convert to PyTorch tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Update Critic networks
        with torch.no_grad(): # No gradient calculation for target
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic_target1(next_state, next_action)
            target_q2 = self.critic_target2(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * (torch.min(target_q1, target_q2) - self.alpha * next_log_prob)

        # Current Q estimates
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        #? Compute Critic losses: Why not target_q.detach()?
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        # Optimize the Critic1 network
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Optimize the Critic2 network
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update Actor network

        # Get new actions and log probabilities for the current states 
        new_action, log_prob = self.actor.sample(state)
        
        # Q values for the new actions 
        q1_new_action = self.critic1(state, new_action)
        q2_new_action = self.critic2(state, new_action)
        
        # Minimum Q value for the new actions
        q_new_action = torch.min(q1_new_action, q2_new_action)

        # Compute Actor loss
        actor_loss = (self.alpha * log_prob - q_new_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.critic1, self.critic_target1)
        self.soft_update(self.critic2, self.critic_target2)
    
        # Clear memory cache just in case
        gc.collect()

        

        
#--------- Test the SAC Agent and choose_action using dummy data ---------#

if __name__ == "__main__":
    env = gym.make("Pendulum-v1") # Create environment: Testing gymnasium's Pendulum-v1
    state_dim = env.observation_space.shape[0] # State dimension
    action_dim = env.action_space.shape[0] # Action dimension
    max_action = float(env.action_space.high[0]) # Maximum action value

    agent = SACAgent(state_dim, action_dim, max_action=max_action) # Initialize SAC agent

    state, _ = env.reset() # Reset environment
    action = agent.choose_action(state) # Choose action using the agent
    print("Chosen action:", action) # Print chosen action