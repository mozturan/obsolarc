from typing import Any
import gymnasium as gym
import numpy as np
from abc import ABC
from gymnasium import spaces

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
            action_dim = 1
            if isinstance(env.action_space, spaces.Box):
                action_dim = env.action_space.shape[0] * self.stack_size
            elif isinstance(env.action_space, spaces.Discrete):
                action_dim = env.action_space.n * self.stack_size
            
            new_observation_dim += action_dim
            
        # 4. Add dimensions for info keys (assuming they are scalar floats)
        new_observation_dim += len(self.info_keys)

        if isinstance(org_observation_shape, tuple):

            self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(new_observation_dim,), 
            dtype=np.float32)
            
        else:
            raise ValueError(
                "env.observation_space.shape is not a tuple. " \
                "Make sure your environment uses continuous states.")
