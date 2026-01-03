import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import pybullet as p

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        
        # Close any existing PyBullet connections first
        try:
            p.disconnect()
        except:
            pass
        
        self.render_mode = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action space: 3 continuous values for x, y, z velocities
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        # Define observation space: 6 values (pipette x,y,z + goal x,y,z)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        self.steps = 0
        self.goal_position = None
        self.robot_id = None  # Will be set dynamically

    def _get_pipette_position(self, observation):
        """Helper function to extract pipette position from observation"""
        # Always get the current robot key from the observation
        robot_id = list(observation.keys())[0]
        
        return np.array(
            observation[robot_id]['pipette_position'],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Set a random goal position within the working area
        self.goal_position = np.array([
            np.random.uniform(-0.187, 0.253),  # x range
            np.random.uniform(-0.1705, 0.2195),   # y range
            np.random.uniform(0.17, 0.2895)    # z range
        ], dtype=np.float32)
        
        # Reset simulation
        self.sim.reset(num_agents=1)
        
        # Get initial state by running one step with zero action
        observation = self.sim.run([[0, 0, 0, 0]], num_steps=1)
        
        # Extract pipette position using helper function
        pipette_pos = self._get_pipette_position(observation)
        
        # Combine pipette position and goal position
        observation = np.concatenate([pipette_pos, self.goal_position])

        self.steps = 0

        return observation, {}

    def step(self, action):
        action_with_drop = [action[0], action[1], action[2], 0]
        observation = self.sim.run([action_with_drop], num_steps=1)

        pipette_pos = self._get_pipette_position(observation)
        observation = np.concatenate([pipette_pos, self.goal_position])

        distance = np.linalg.norm(pipette_pos - self.goal_position)
        reward = float(-distance)  # Convert to Python float
        
        threshold = 0.001
        if distance < threshold:
            terminated = True
            reward += 100.0  # Bonus as float
        else:
            terminated = False

        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {'distance': float(distance)}  # Also convert distance in info
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self):
        pass
    
    def close(self):
        self.sim.close()