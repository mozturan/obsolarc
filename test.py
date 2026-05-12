import gymnasium as gym # Gymnasium for environments
from gymnasium.spaces import Box
import numpy as np
import highway_env

# config={
#                             "observation": {
#                                 "type": "Kinematics",
#                                 "vehicles_count": 1,
#                                 "features": ["presence", 
#                                              "x", "y", 
#                                              "vx", "vy", 
#                                              "cos_h", "sin_h",
#                                              "heading", "long_off",
#                                              "lat_off", "ang_off"],
#                                 "features_range": {
#                                     "x": [-100, 100],
#                                     "y": [-100, 100],
#                                     "vx": [-20, 20],
#                                     "vy": [-20, 20]
#                                 },        
#                                 "absolute": False,
#                                 "order": "sorted"
#                                             },
#                             "action": {
#                                 "type": "ContinuousAction",
#                                 "longitudinal": True,
#                                 "lateral": True
#                             },
#                             "simulation_frequency": 15,
#                             "policy_frequency": 5,
#                             "duration": 300,
#                             "collision_reward": -1,
#                             "lane_centering_cost": 4,
#                             "action_reward": -0.3,
#                             "controlled_vehicles": 1,
#                             "other_vehicles": 0,
#                             "screen_width": 600,
#                             "screen_height": 600,
#                             "centering_position": [0.5, 0.5],
#                             "scaling": 7,
#                             "show_trajectories": True,
#                             "render_agent": True,
#                             "offscreen_rendering": False
#                         }

env = gym.make('racetrack-v0', render_mode='human')


obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = [1.0]
    obs, reward, done, truncated, info = env.step(action)
