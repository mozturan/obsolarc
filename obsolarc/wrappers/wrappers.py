from typing import Any
import gymnasium as gym
import numpy as np
from abc import ABC
from gymnasium import spaces
from collections import deque


class Wrapper(ABC, gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, float(reward), terminated, truncated, info
    
class SACHighwayWrapper(Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        original_obs_shape = env.observation_space.shape
        if isinstance(original_obs_shape, tuple):
            new_obs_shape = (int(np.prod(original_obs_shape)) + 1,) # Flatten the original observation and add 1 dimension for speed
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=new_obs_shape, dtype=np.float32)
        else:
            raise ValueError("env.observation_space.shape is not a tuple. Make sure your environment uses continuous states.")
                                
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(obs, info), float(reward), terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        state, info = super().reset(seed=seed, options=options)
        return self._observation(state, info), info
    
    def _getinfo(self, info: dict[str, Any]) -> np.ndarray:
        return np.array(float(info["speed"]))

    def _observation(self, obs: Any, info: Any) -> np.ndarray:
        
        "! This function takes the original observation and info, "
        "flattens the observation, extracts the speed from info, "
        "and concatenates them into a single observation array."

        info = self._getinfo(info)
        obs = obs.flatten() if isinstance(obs, np.ndarray) else np.array(obs).flatten()
        obs = np.concatenate((obs, [info]), axis=0)
        return obs

class MultiFeatureWrapper(Wrapper):
    def __init__(self, env, stack_size: int =2, include_prev_action: bool =True, info_keys=None):
        super().__init__(env)
        self.stack_size = stack_size
        self.include_prev_action = include_prev_action
        self.info_keys = info_keys or []

        # Initialize deques to hold past observations and actions
        self._obs_buffer = deque([], maxlen=stack_size)
        self._action_buffer = deque([], maxlen=stack_size)

        self._obs_reshaper(env=env)
        
    def _obs_reshaper(self, env) -> None:

        # 1. Start with the original observation shape
        # Assuming a Box space for vector observations
        org_observation_shape = env.observation_space.shape
        observation_shape = int(np.prod(org_observation_shape))

        # 2. Calculate the stacked dimension
        # New shape = (Original * Stack Count)
        new_observation_dim = observation_shape * self.stack_size
        
        # 3. Add dimension for previous action(s)
        if self.include_prev_action:
            if isinstance(env.action_space, spaces.Box):
                action_dim = env.action_space.shape[0]
            elif isinstance(env.action_space, spaces.Discrete):
                action_dim = env.action_space.n
            else: 
                raise ValueError(f"Unsupported action space: {type(env.action_space)}")
            new_observation_dim += action_dim * self.stack_size
            
        # 4. Add dimensions for info keys (assuming they are scalar floats)
        new_observation_dim += len(self.info_keys)

        if isinstance(org_observation_shape, tuple):

            self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(new_observation_dim,), 
            dtype=np.float32)

        else: #! This error needs to go to top actually
            raise ValueError(
                "env.observation_space.shape is not a tuple. " \
                "Make sure your environment uses continuous states.")

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        
        self._obs_buffer.append(self._flatten(obs))
        if self.include_prev_action:
            self._action_buffer.append(self._encode_action(action))

        new_obs = self._observation(obs, info)
        return new_obs, float(reward), terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
    
        #Clearing buffers
        self._obs_buffer.clear()
        self._action_buffer.clear()

        for _ in range(self.stack_size):
            self._obs_buffer.append(self._flatten(obs))

        if self.include_prev_action:
            if isinstance(self.env.action_space, spaces.Box):
                zero_action = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
            elif isinstance(self.env.action_space, spaces.Discrete):
                zero_action = 0
            else: 
                raise ValueError(f"Unsupported action space: {type(self.env.action_space)}")

            for _ in range(self.stack_size):
                self._action_buffer.append(self._encode_action(zero_action))

        return self._observation(obs, info), info
    
    def _getinfo(self, info: dict[str, Any]) -> np.ndarray:

        values = []
        for key in self.info_keys:
            value = info.get(key, 0.0)
            value = self._flatten(value)

            if len(value) !=1:
                raise ValueError(f"Info key '{key}' must be scalar, got {value}")

            values.append(float(value))
        return np.array(values, dtype=np.float32)
    
    def _observation(self, obs: Any, info: Any) -> np.ndarray:

        obs_stack = np.concatenate(self._obs_buffer, axis=0)

        if self.include_prev_action:
            action_stack = np.concatenate(self._action_buffer, axis=0)
            obs_stack = np.concatenate([obs_stack, action_stack], axis=0)

        if self.info_keys:
            values = self._getinfo(info = info)
            obs_stack = np.concatenate([obs_stack, values], axis = 0)
        
        return obs_stack.astype(np.float32)

    def _flatten(self, x: Any) -> np.ndarray:

        if isinstance(x, np.ndarray):
            return x.flatten()
        return np.array([x]).flatten()
    
    def _encode_action(self, action: Any) -> np.ndarray:
        if isinstance(self.env.action_space, spaces.Discrete):
            encoded = np.zeros(self.env.action_space.n, dtype=np.float32)
            encoded[action] = 1.0
            return encoded
        else:
            return np.array(action).flatten()