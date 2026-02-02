# A SAC implementation using PyTorch for a simple continuous control task

import time
import gymnasium as gym # Gymnasium for environments
from gymnasium.spaces import Box

import gc
from click import Tuple
import torch # PyTorch library
import torch.nn as nn # Neural network module
import torch.optim as optim # Optimization algorithms
import numpy as np # NumPy for numerical operations
from buffer import ReplayBuffer # Import ReplayBuffer from buffer.py

# Define the Actor network
class Actor(nn.Module):

    # Initialize the Actor network
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float):
        super(Actor, self).__init__() #?? Initialize parent class
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    # Define the forward pass
    def forward(self, state: torch.Tensor):
        # check for NaNs in input state
        if torch.isnan(state).any():
            print("NaN detected in input state!")
            print("Input state:", state)

        # check for infinities in input state
        if torch.isinf(state).any():
            print("Infinity detected in input state!")
            print("Input state:", state)

        x = torch.relu(self.fc1(state))
        if torch.isnan(x).any():
            print("NaN detected after fc1!")
            print("Output after fc1:", x)


        x = torch.relu(self.fc2(x))
        if torch.isnan(x).any():
            print("NaN detected after fc2!")
            print("Output after fc2:", x)


        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)  #! Limit log_std to avoid numerical issues
        std = torch.exp(log_std)    
        return mu, std # Return mean and standard deviation of action distribution
    
    def sample(self, state: torch.Tensor) : 
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()  # Reparameterization trick
        action_tanh = torch.tanh(z) # Tanh squashing
        action = action_tanh * self.max_action  # Scale action to environment's action range
        
        #! Look for correct log prob calculation 
        log_prob = normal.log_prob(z)
        log_prob_minus= torch.log(1 - action_tanh.pow(2) + 1e-6) # Correction for Tanh squashing
        log_prob = log_prob - log_prob_minus
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob # Return sampled action and log probability
    
# Define the Critic network
class Critic(nn.Module):

    # Initialize the Critic network
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(Critic, self).__init__() # ?? Initialize parent class
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)

    # Define the forward pass
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        q_value = self.q_head(x)
        return q_value
    
# Soft Actor-Critic (SAC) Agent
class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, 
                 buffer_size: int=int(1e6), min_buffer_size: int=1000,
                 batch_size: int=256, hidden_dim: int=256, max_action: float=1.0,
                 actor_lr: float=3e-4, critic_lr: float=3e-4,
                 gamma: float=0.99, tau: float=0.005, alpha: float=0.2):

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
    def choose_action(self, state: np.ndarray, evaluate: bool=False) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device) # Convert state to tensor and move to device
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        
        if evaluate:
            #! This can be optimized further: refactor to avoid code duplication
            mu, _ = self.actor.forward(state_tensor)
            action = torch.tanh(mu) * self.actor.max_action
        else:
            action, _ = self.actor.sample(state_tensor)

        return action.squeeze(0).cpu().data.numpy() # Remove batch dimension and convert to numpy array
        
    # Soft update target networks
    def soft_update(self, net: torch.nn.Module, target_net: torch.nn.Module) -> None:
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # Update the SAC agent
    def train(self) -> None:
        
        if self.replay_buffer.size < self.min_buffer_size:
            return  # Not enough data to train
        
        # Sample a batch of transitions from the replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample_buffer(self.batch_size)

        # Convert to PyTorch tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).view(-1, 1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)

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
        #*  it is already detached from the gradient graph. Adding .detach() is redundant but safe.
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
        # torch.cuda.empty_cache() #? Does this work? 


#--------- Test the SAC Agent and choose_action using dummy data ---------#

if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0", render_mode="human", 
                   goal_velocity=1.0) # Create environment: Testing gymnasium's Pendulum-v1
    
    # print("obs space:", env.observation_space)  # should be a Box with shape (3,)
    # print("act space:", env.action_space)       # should be a Box with high ≈ [2.]

    # Get state and action dimensions
    state_space = env.observation_space.shape # (3,)
    if isinstance(state_space, tuple):
        state_shape = int(state_space[0])  # Adjust index as necessary
    else:
        raise ValueError("env.observation_space.shape is not a tuple. Make sure your environment uses continuous states.")

    action_space = env.action_space.shape # (1,)
    if isinstance(action_space, tuple):
        action_shape = int(action_space[0])  # Adjust index as necessary
    else:
        raise ValueError("env.action_space.shape is not a tuple. Make sure your environment uses continuous actions.")

    # Get maximum action value
    if isinstance(env.action_space, Box):
        max_action = float(env.action_space.high[0])  # Maximum action value
    else:
        raise ValueError("env.action_space is not a Box space. Make sure your environment uses continuous actions.")

    agent = SACAgent(state_shape, action_shape, max_action=max_action) # Initialize SAC agent

    state, _ = env.reset() # Reset environment
    action = agent.choose_action(state) # Choose action using the agent
    print("Chosen action:", action) # Print chosen action

    for i in range(50000): # Run for 5 steps
        next_state, reward, terminated, truncated, info = env.step(action) # Take action in environment
        done = terminated or truncated
        agent.replay_buffer.store_transition(state, action, float(reward), next_state, done) # Store transition in replay buffer
        state = next_state # Update state
        action = agent.choose_action(state) # Choose next action
        agent.train() # Train the agent
        print(f"Step {i+1} completed. Reward: {reward}")
        # if done:
        #     state, _ = env.reset() # Reset environment if done
        #     print("Environment reset.")
        #     time.sleep(5)
        


    env.close()