"""
Agent module for Network Routing.
Provides Tabular Q-Learning Agent (QLearningAgent) and PyTorch Deep Q-Network Agent (DQNAgent, DQNModel, ReplayBuffer).
"""

import collections
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QLearningAgent:
    """
    Tabular Q-Learning agent with continuous state quantization, action masking,
    and JSON serialization support.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ) -> None:
        """
        Initialize Q-Learning agent hyperparameters and Q-table.

        Args:
            num_states: Discrete bin count for continuous state quantization.
            num_actions: Number of available actions in the environment.
            learning_rate: Alpha learning rate hyperparameter.
            discount_factor: Gamma discount factor for future rewards.
            epsilon: Initial exploration probability.
            epsilon_decay: Multiplicative decay factor per step/episode.
            min_epsilon: Lower bound threshold for epsilon.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Internal Q-table mapping discrete tuple state -> np.ndarray Q-values
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float64)
        )

    def quantize_state(self, state: Any) -> Tuple:
        """
        Convert continuous observation array or structure into discrete tuple representation.

        Args:
            state: Continuous vector, dict, or scalar state representation.

        Returns:
            Hashable tuple key suitable for Q-table lookup.
        """
        if isinstance(state, tuple):
            return state
        if isinstance(state, (int, np.integer)):
            return (int(state),)
        if isinstance(state, dict):
            return tuple(sorted((k, self.quantize_state(v)) for k, v in state.items()))
        if isinstance(state, (list, np.ndarray)):
            state_arr = np.asarray(state, dtype=np.float64)
            quantized = []
            for val in state_arr.flat:
                if val.is_integer():
                    quantized.append(int(val))
                else:
                    if 0.0 <= val <= 1.0:
                        binned = int(np.clip(val * self.num_states, 0, self.num_states - 1))
                    else:
                        binned = int(np.clip(val, 0, 1000))
                    quantized.append(binned)
            return tuple(quantized)
        return (str(state),)

    def get_q_values(self, state: Any) -> np.ndarray:
        """
        Get Q-values vector for a given state.
        """
        q_key = self.quantize_state(state)
        return self.q_table[q_key]

    def choose_action(
        self,
        state: Any,
        valid_actions: Optional[Union[List[int], np.ndarray, Set[int], Tuple[int, ...]]] = None,
    ) -> int:
        """
        Choose action using epsilon-greedy policy with optional action masking.

        Args:
            state: Environment state observation.
            valid_actions: List of valid action indices or boolean action mask array.

        Returns:
            Selected action index (int).
        """
        q_key = self.quantize_state(state)

        # Parse valid action indices
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray) and valid_actions.dtype == bool:
                valid_indices = np.where(valid_actions)[0]
            elif isinstance(valid_actions, (list, set, tuple, np.ndarray)):
                valid_indices = np.array(list(valid_actions), dtype=int)
            else:
                valid_indices = np.arange(self.num_actions)
        else:
            valid_indices = np.arange(self.num_actions)

        if len(valid_indices) == 0:
            valid_indices = np.arange(self.num_actions)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Exploration
            return int(np.random.choice(valid_indices))
        else:
            # Exploitation: select action with maximum Q-value among valid actions
            q_values = self.q_table[q_key]
            valid_q = q_values[valid_indices]
            max_q = np.max(valid_q)
            best_mask = np.isclose(valid_q, max_q)
            best_actions = valid_indices[best_mask]
            return int(np.random.choice(best_actions))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """
        Update Q-table value using the standard Bellman equation and decay epsilon.

        Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode terminated.

        Returns:
            TD error float value.
        """
        s_key = self.quantize_state(state)
        s_prime_key = self.quantize_state(next_state)

        q_s = self.q_table[s_key]
        q_s_prime = self.q_table[s_prime_key]

        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(q_s_prime)

        td_error = target - q_s[action]
        q_s[action] += self.learning_rate * td_error

        # Decay epsilon down to min_epsilon threshold
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return float(td_error)

    def save_q_table(self, filepath: str) -> None:
        """
        Serialize Q-table dictionary to JSON file format.

        Args:
            filepath: Destination file path.
        """
        serialized = {}
        for key, q_vals in self.q_table.items():
            key_str = json.dumps(list(key))
            serialized[key_str] = q_vals.tolist()

        dir_name = os.path.dirname(os.path.abspath(filepath))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    def load_q_table(self, filepath: str) -> None:
        """
        Deserialize Q-table dictionary from JSON file format.

        Args:
            filepath: Source JSON file path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Q-table file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            serialized = json.load(f)

        self.q_table.clear()
        for key_str, q_vals_list in serialized.items():
            key_tuple = tuple(json.loads(key_str))
            self.q_table[key_tuple] = np.array(q_vals_list, dtype=np.float64)


class ReplayBuffer:
    """
    Experience Replay Buffer storing transition tuples using fixed-size deque.
    """

    def __init__(self, capacity: int = 10000) -> None:
        """
        Initialize replay buffer with maximum capacity.

        Args:
            capacity: Maximum number of experience tuples to retain.
        """
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """
        Append experience tuple to buffer.

        Args:
            state: State vector or observation.
            action: Action index.
            reward: Scalar reward received.
            next_state: Next state vector or observation.
            done: Episode termination boolean indicator.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample random minibatch of experiences converted to PyTorch FloatTensors.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) PyTorch Tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states, dtype=np.float32), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        """
        Get current size of replay buffer.
        """
        return len(self.buffer)


class DQNModel(nn.Module):
    """
    Lightweight PyTorch MLP for mapping state vector to action Q-values.
    Sub-millisecond CPU execution architecture.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        Initialize MLP network layers.

        Args:
            state_dim: Input dimension of state feature vector.
            action_dim: Output dimension (number of discrete actions).
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values vector for input batch x.
        """
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for continuous observation spaces and discrete action selection.
    Uses experience replay, target network stabilization, Huber loss, and action masking.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize DQNAgent neural networks, target network, optimizer, and replay buffer.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

        self.policy_net = DQNModel(state_dim, action_dim)
        self.target_net = DQNModel(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def choose_action(
        self,
        state: Any,
        valid_actions: Optional[Union[List[int], np.ndarray, Set[int], Tuple[int, ...]]] = None,
    ) -> int:
        """
        Select action using epsilon-greedy policy with optional action masking.
        Inference is wrapped in torch.no_grad() for maximum speed.

        Args:
            state: Continuous state vector or observation array.
            valid_actions: List of valid action indices or boolean action mask array.

        Returns:
            Selected action index (int).
        """
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray) and valid_actions.dtype == bool:
                valid_indices = np.where(valid_actions)[0]
            elif isinstance(valid_actions, (list, set, tuple, np.ndarray)):
                valid_indices = np.array(list(valid_actions), dtype=int)
            else:
                valid_indices = np.arange(self.action_dim)
        else:
            valid_indices = np.arange(self.action_dim)

        if len(valid_indices) == 0:
            valid_indices = np.arange(self.action_dim)

        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_indices))

        state_t = torch.tensor(np.array(state, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0).numpy()

        valid_q = q_values[valid_indices]
        max_q = np.max(valid_q)
        best_mask = np.isclose(valid_q, max_q)
        best_actions = valid_indices[best_mask]

        return int(np.random.choice(best_actions))

    def update(self) -> Optional[float]:
        """
        Sample minibatch from ReplayBuffer, compute Smooth L1 (Huber) loss against target network,
        perform backpropagation, update parameters, and decay epsilon.

        Returns:
            Computed scalar loss float value, or None if buffer size < batch_size.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute Q(s, a) using policy_net
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute max_a' Q_target(s', a') using target_net
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(dim=1)[0]
            expected_state_action_values = rewards + (1.0 - dones) * self.gamma * next_state_values

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def update_target_network(self) -> None:
        """
        Copy parameters from policy_net to target_net.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath: str) -> None:
        """
        Save PyTorch policy network state dictionary to file.
        """
        dir_name = os.path.dirname(os.path.abspath(filepath))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load PyTorch policy network state dictionary from file and update target network.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        try:
            state_dict = torch.load(filepath, weights_only=True)
        except Exception:
            state_dict = torch.load(filepath)
        self.policy_net.load_state_dict(state_dict)
        self.update_target_network()
