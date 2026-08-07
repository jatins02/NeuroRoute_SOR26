"""
Unit test for NetworkRoutingEnv.
"""

import gymnasium as gym
import numpy as np

from neuroroute.ai.env import NetworkRoutingEnv

def test_env_initialization():
    """Test that the environment can be initialized and spaces inspected."""
    
    # Create environment
    env = NetworkRoutingEnv(num_nodes=5)
    
    # Check action space
    assert env.action_space.n == 5
    assert isinstance(env.action_space, gym.spaces.Discrete)
    
    # Check observation space
    assert env.observation_space.shape == (11,)  # 2 * num_nodes + 1
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32
    
    # Reset environment
    obs, info = env.reset()
    
    # Check observation shape
    assert obs.shape == (11,)
    assert np.all(obs >= env.observation_space.low) and np.all(obs <= env.observation_space.high)
    
    # Check info
    assert "current_node" in info
    assert "destination_node" in info
    
    # Test step
    action = 0  # Send to node 0
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check step returns correct types
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    print("✅ All tests passed!")
    
    env.close()

if __name__ == "__main__":
    test_env_initialization()