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

        # self.config = config or {}
        
        # Initialize the environment with the given configuration
        #! In this case i can't set new obs shape correctly idk why
        # self._init_env()

        
        original_obs_shape = env.observation_space.shape
        if isinstance(original_obs_shape, tuple):
            new_obs_shape = (int(np.prod(original_obs_shape)) + 1,) # Flatten the original observation and add 1 dimension for speed
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=new_obs_shape, dtype=np.float32)
        else:
            raise ValueError("env.observation_space.shape is not a tuple. Make sure your environment uses continuous states.")
                                
    # def _init_env(self):
    #     # Update the environment's configuration with the provided config
    #     config = getattr(self.env.unwrapped, 'config', None)
    #     if config is not None:
    #         config.update(self.config)
    #     else:
    #         raise ValueError("The environment does not have a 'config' attribute. Make sure your environment is compatible with this wrapper.")

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

class MultiFeatureWrapper(gym.ObservationWrapper):
    def __init__(self, env, stack_size=4, include_prev_action=True, info_keys=None):
        super().__init__(env)
        self.stack_size = stack_size
        self.include_prev_action = include_prev_action
        self.info_keys = info_keys or []
        
        # 1. Start with the original observation shape
        # Assuming a Box space for vector observations
        orig_shape = env.observation_space.shape[0]
        
        # 2. Calculate the stacked dimension
        # New shape = (Original * Stack Count)
        new_dim = orig_shape * self.stack_size
        
        # 3. Add dimension for previous action
        if self.include_prev_action:
            action_dim = 1
            if isinstance(env.action_space, spaces.Box):
                action_dim = env.action_space.shape[0]
            elif isinstance(env.action_space, spaces.Discrete):
                action_dim = 1 # Or use env.action_space.n if one-hot encoding
            
            new_dim += action_dim
            
        # 4. Add dimensions for info keys (assuming they are scalar floats)
        new_dim += len(self.info_keys)
        
        # 5. Update the space!
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(new_dim,), 
            dtype=np.float32
        )