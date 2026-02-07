import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNetwork(nn.Module, ABC):
    def __init__(self):
        super(BaseNetwork, self).__init__()

class BasePolicy(BaseNetwork):
    def __init__(self,                  
                 state_dim: int, 
                 action_dim: int, 
                 hidden_sizes: list, 
                 max_action: float):
        super().__init__()

    @staticmethod
    def initialize_weights(module: nn.Module):
        pass
    
    @abstractmethod
    def sample(self, state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def forward(self, state)-> tuple[torch.Tensor, torch.Tensor]:
        pass

class BaseValueFunction(BaseNetwork):
    def __init__(self, state_dim: int, 
                 action_dim: int, 
                 hidden_sizes: list[int]):
        super().__init__()

    @staticmethod
    def initialize_weights(module: nn.Module):
        pass
