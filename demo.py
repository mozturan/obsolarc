# A SAC implementation using PyTorch for a simple continuous control task

import torch # PyTorch library
import torch.nn as nn # Neural network module
import torch.optim as optim # Optimization algorithms
import gymnasium as gym # Gymnasium for environments
import numpy as np # NumPy for numerical operations

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
        return action, log_prob
    
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
                 hidden_dim=256, max_action=1.0,
                 actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.critic_target1.load_state_dict(self.critic1.state_dict()) # Copy weights to target
        self.critic_target2.load_state_dict(self.critic2.state_dict()) # Copy weights to target

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