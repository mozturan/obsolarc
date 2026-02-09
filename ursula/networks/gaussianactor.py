import torch
import torch.nn as nn
from .base import BasePolicy

class GaussianActor(BasePolicy):
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 hidden_sizes: list, 
                 max_action: float,
                 log_std_min: float = -20.0, 
                 log_std_max: float = 2.0):
        super().__init__(observation_space, 
                        action_space,
                        hidden_sizes, 
                        max_action)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = max_action

        # Build the network architecture
        layers = []
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        
        # Output layers for mean and log standard deviation
        self.mu = nn.Linear(input_dim, action_dim)
        self.sigma = nn.Linear(input_dim, action_dim)

        # Combine the layers into a sequential model
        self.network = nn.Sequential(*layers)
        # self.apply(self.initialize_weights) # Initialize weights using the method from BasePolicy

    def forward(self, 
                state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = self.network(state)
        mean = self.mu(x)
        log_std = self.sigma(x)
        log_std = torch.clamp(log_std, 
                              self.log_std_min, 
                              self.log_std_max)
        std = torch.exp(log_std)    

        return mean, std

    def sample(self, state)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.forward(state)
        normal_dist = torch.distributions.Normal(mean, std)
        x = normal_dist.rsample()  # Reparameterization trick
        y = torch.tanh(x) # Squash the output to be in [-1, 1]
        action = y * self.max_action
        
        log_prob = normal_dist.log_prob(x) # Sum over action dimensions
        log_prob_minus = torch.log(1 - y.pow(2) + 1e-6)  # Adjust for the tanh squashing
        log_prob = log_prob - log_prob_minus # Sum over action dimensions
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean = torch.tanh(mean) * self.max_action # Squash mean to be in [-max_action, max_action]
        return action, log_prob, mean