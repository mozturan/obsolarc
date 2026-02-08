import gymnasium as gym # Gymnasium for environments
from gymnasium.spaces import Box
import numpy as np
from ursula.sac import SAC
import highway_env
from wrapper import SACHighwayWrapper

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
    env = SACHighwayWrapper(env)

    # print(env.unwrapped.config)

    # Get state and action dimensions
    state_space = env.observation_space.shape # (3,)
    if isinstance(state_space, tuple):
        state_shape = int(np.prod(state_space)) 

    else:
        raise ValueError("env.observation_space.shape is not a tuple. Make sure your environment uses continuous states.")

    action_space = env.action_space.shape # (1,)
    if isinstance(action_space, tuple):
        action_shape = int(np.prod(action_space))
    else:
        raise ValueError("env.action_space.shape is not a tuple. Make sure your environment uses continuous actions.")

    # Get maximum action value
    if isinstance(env.action_space, Box):
        max_action = float(env.action_space.high[0])  # Maximum action value
    else:
        raise ValueError("env.action_space is not a Box space. Make sure your environment uses continuous actions.")

    # agent = SACAgent(state_shape, action_shape, max_action=max_action) # Initialize SAC agent
    agent = SAC(state_shape, 
                     action_shape, 
                     max_action=1.0,
                     critic_lr=0.003,
                     actor_lr=0.003,
                     batch_size=64,
                     min_buffer_size=100,
                     hidden_sizes=[256, 256],
                     auto_entropy=True) # Initialize SAC agent

    state, _ = env.reset() # Reset environment
    action = agent.choose_action(state) # Choose action using the agent

    print(state)
    print(state.shape)

    for i in range(50000): # Run for 5 steps
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
        print(f"Step {i+1}")
        if done:
            state, _ = env.reset() # Reset environment if done

    env.close()