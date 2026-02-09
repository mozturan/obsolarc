import torch
import torch.nn as nn
from .base import BaseValueFunction

class Critic(BaseValueFunction):

    def __init__(self, 
                 observation_space, 
                 action_space, 
                 hidden_sizes: list = [256, 256]):
        
        super().__init__(observation_space, 
                         action_space, 
                         hidden_sizes)

        self.state_dim = observation_space.shape[0] #if hasattr(observation_space, 'shape') else observation_space.n
        self.action_dim = action_space.shape[0] #if hasattr(action_space, 'shape') else action_space.n

        # Build the network architecture
        layers = []
        input_dim = self.state_dim + self.action_dim  # Concatenate state and action
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        
        # Output layer for Q-value
        layers.append(nn.Linear(input_dim, 1))

        # Combine the layers into a sequential model
        self.network = nn.Sequential(*layers)
        # self.apply(self.initialize_weights) # Initialize weights using the method from BasePolicy

    def forward(self, 
                state: torch.Tensor, 
                action: torch.Tensor) -> torch.Tensor:
        
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action
        q_value = self.network(x)
        return q_value