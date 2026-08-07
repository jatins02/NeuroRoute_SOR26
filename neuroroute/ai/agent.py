"""
Tabular Q-Learning Agent for Network Routing.
"""

import json
import os
from typing import Optional, Union, List, Tuple, Dict, Any, Set
from collections import defaultdict
import numpy as np


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
