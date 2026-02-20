import gymnasium as gym # Gymnasium for environments
from gymnasium.spaces import Box
import numpy as np
from obsolarc.sac import SAC
import highway_env
from obsolarc.wrappers.wrappers import SACHighwayWrapper, StackWrapper

if __name__ == "__main__":

    config={
                            "observation": {
                                "type": "Kinematics",
                                "vehicles_count": 1,
                                "features": ["presence", 
                                             "x", "y", 
                                             "vx", "vy", 
                                             "cos_h", "sin_h",
                                             "heading", "long_off",
                                             "lat_off", "ang_off"],
                                             "grid_size": [[-18, 18], [-18, 18]],
                                "features_range": {
                                    "x": [-100, 100],
                                    "y": [-100, 100],
                                    "vx": [-20, 20],
                                    "vy": [-20, 20]
                                },                            },
                            "action": {
                                "type": "ContinuousAction",
                                "longitudinal": True,
                                "lateral": True
                            },
                            "simulation_frequency": 15,
                            "policy_frequency": 5,
                            "duration": 300,
                            "collision_reward": -1,
                            "lane_centering_cost": 4,
                            "action_reward": -0.3,
                            "controlled_vehicles": 1,
                            "other_vehicles": 0,
                            "screen_width": 600,
                            "screen_height": 600,
                            "centering_position": [0.5, 0.5],
                            "scaling": 7,
                            "show_trajectories": True,
                            "render_agent": True,
                            "offscreen_rendering": False
                        }

    env = gym.make('racetrack-v0', render_mode='human', config=config)
    env = StackWrapper(env, stack_size=2, include_prev_action=True, info_keys=["speed"])

    # agent = SACAgent(state_shape, action_shape, max_action=max_action) # Initialize SAC agent
    agent = SAC(     env=env, 
                     max_action=env.max_action,
                     critic_lr=0.003,
                     actor_lr=0.003,
                     batch_size=256,
                     min_buffer_size=500,
                     hidden_sizes=[512, 512],
                     auto_entropy=True) # Initialize SAC agent

    episodes = 5000
    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state) # Choose action using the agent
            next_state, reward, terminated, truncated, info = env.step(action) # Take action in environment
            done = terminated or truncated or info.get('on_road_reward', 0)
            agent.replay_buffer.store_transition(state, 
                                                action, 
                                                float(reward), 
                                                next_state, 
                                                done) # Store transition in replay buffer
            state = next_state # Update state
            action = agent.choose_action(state) # Choose next action
            agent.train() # Train the agent        
        
        print("Episode : ", episode)
    
    env.close()