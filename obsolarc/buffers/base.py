from typing import Tuple
import numpy as np
import random as rndm
from abc import ABC, abstractmethod

class Buffer(ABC):
    """
    Abstract class for replay buffer
    """

    def __init__(self, 
                 max_size: int, 
                 state_dim: int, 
                 action_dim: int):
        pass


    @abstractmethod
    def store_transition(self, 
                         state: np.ndarray, 
                         action: np.ndarray, 
                         reward: float, 
                         state_: np.ndarray, 
                         done: bool) -> None:
        
        """
        Store a new transition
        """
        pass

    @abstractmethod
    def sample_buffer(self, 
                      batch_size:int, 
                      priority_scale:float=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions from the buffer
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of stored transitions
        """
        pass
