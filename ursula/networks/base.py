import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNetwork(nn.Module, ABC):

    """
    It handles boilerplate like weight initialization, saving/loading, and device management.
    """
    def __init__(self):
        super(BaseNetwork, self).__init__()

class BasePolicy(BaseNetwork):

    """
    Defines the API for .predict() and .forward().
    """

    def __init__(self,                  
                 observation_space, 
                 action_space, 
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

    """
        Defines the API for .forward().
    """
    
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 hidden_sizes: list[int]):
        super().__init__()

    @staticmethod
    def initialize_weights(module: nn.Module):
        pass
