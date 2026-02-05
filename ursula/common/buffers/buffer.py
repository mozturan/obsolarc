from typing import Tuple
import numpy as np
import random as rndm
from abc import ABC, abstractmethod

class AbstractReplayBuffer(ABC):
    """
    Abstract class for replay buffer
    """
    @abstractmethod
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, state_: np.ndarray, done: bool) -> None:
        """
        Store a new transition
        """
        pass

    @abstractmethod
    def sample_buffer(self, batch_size:int, priority_scale:float=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

class ReplayBuffer(AbstractReplayBuffer):
    """
    A simple replay buffer for storing and sampling transitions

    """
    def __init__(self, max_size: int, state_dim: int, action_dim: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory for the buffer
        self.state_memory = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_state_memory = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.done_memory = np.zeros((self.max_size, 1), dtype=bool)
    
    # Store a new transition in the buffer
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, state_: np.ndarray, done: bool) -> None:
        self.state_memory[self.ptr] = state.reshape(-1)
        self.action_memory[self.ptr] = action.reshape(-1)
        self.reward_memory[self.ptr] = reward
        self.next_state_memory[self.ptr] = state_.reshape(-1)
        self.done_memory[self.ptr] = done

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_buffer(self, batch_size: int, priority_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_mem = self.size
        batch_indices = rndm.sample(range(max_mem), batch_size)
        # batch_indices = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]
        next_states = self.next_state_memory[batch_indices]
        dones = self.done_memory[batch_indices]

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size
    
    def save(self, filename: str) -> None:
        np.savez_compressed(
            filename,
            state_memory=self.state_memory,
            action_memory=self.action_memory,
            reward_memory=self.reward_memory,
            next_state_memory=self.next_state_memory,
            done_memory=self.done_memory,
            ptr=self.ptr,
            size=self.size
        )

    def load(self, filename: str) -> None:
        data = np.load(filename)
        self.state_memory = data['state_memory']
        self.action_memory = data['action_memory']
        self.reward_memory = data['reward_memory']
        self.next_state_memory = data['next_state_memory']
        self.done_memory = data['done_memory']
        self.ptr = data['ptr'].item()
        self.size = data['size'].item()
