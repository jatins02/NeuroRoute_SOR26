"""
Gymnasium environment for network packet routing simulation.
"""

from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NetworkRoutingEnv(gym.Env):
    """
    Custom Gymnasium environment for AI-driven network packet routing.

    Simulates packet traversal across a graph network topology with dynamic
    queue depths, link latencies, action masking, penalties for invalid actions/drops,
    and positive rewards upon reaching the target destination.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        num_nodes: int = 5,
        max_queue_capacity: int = 100,
        topology_graph: Optional[Any] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 100,
    ) -> None:
        """
        Initialize the network routing environment.

        Args:
            num_nodes: Total number of nodes in the network topology.
            max_queue_capacity: Maximum queue depth capacity per node.
            topology_graph: Optional graph specification (matrix, dict, or TopologyManager).
            render_mode: Rendering mode ("human" or "ansi").
            max_steps: Maximum step budget before episode truncation.
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.max_queue_capacity = max_queue_capacity
        self.render_mode = render_mode
        self.topology_graph = topology_graph
        self.max_steps = max_steps

        # Action Space: Choose a neighbor node index to forward packet to (0 to num_nodes - 1)
        self.action_space = spaces.Discrete(num_nodes)

        # Observation Space:
        # 1. Normalized queue lengths per node [0.0, 1.0] (num_nodes)
        # 2. Target destination node ID [0, num_nodes - 1] (1)
        # 3. Adjacent link latencies in ms from current node [0.0, 1000.0] (num_nodes)
        obs_dim = 2 * num_nodes + 1
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32)
        high[:num_nodes] = 1.0
        high[num_nodes] = float(num_nodes - 1)
        high[num_nodes + 1 :] = 1000.0

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

        # Build adjacency matrix and initial link latencies matrix (num_nodes x num_nodes)
        self.adj_matrix, self.link_latencies = self._parse_topology(topology_graph)

        # Environment state variables
        self.current_node: int = 0
        self.destination_node: int = num_nodes - 1
        self.queue_depths: np.ndarray = np.zeros(num_nodes, dtype=np.float32)

        # Performance and state metrics
        self.dropped_packets: int = 0
        self.successful_deliveries: int = 0
        self.total_packets: int = 0
        self.step_count: int = 0

    def _parse_topology(self, topology: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse topology_graph into boolean adjacency matrix and float link latencies matrix.
        """
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
        lat = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        if topology is None:
            # Default topology: ring network with cross links for multi-path routing options
            for i in range(self.num_nodes):
                next_node = (i + 1) % self.num_nodes
                prev_node = (i - 1) % self.num_nodes
                adj[i, next_node] = True
                adj[i, prev_node] = True
                lat[i, next_node] = 10.0
                lat[i, prev_node] = 10.0

                if self.num_nodes >= 4:
                    skip_node = (i + 2) % self.num_nodes
                    adj[i, skip_node] = True
                    lat[i, skip_node] = 25.0
        elif isinstance(topology, np.ndarray):
            adj = (topology > 0) & (~np.eye(self.num_nodes, dtype=bool))
            lat = topology.astype(np.float32)
        elif hasattr(topology, "graph") and isinstance(topology.graph, dict):
            # Support TopologyManager objects from neuroroute.network.topology
            sorted_node_keys = sorted(list(topology.graph.keys()))
            node_map = {n: i for i, n in enumerate(sorted_node_keys[: self.num_nodes])}
            for u, neighbors in topology.graph.items():
                if u not in node_map:
                    continue
                u_idx = node_map[u]
                for v, metrics in neighbors.items():
                    if v not in node_map:
                        continue
                    v_idx = node_map[v]
                    if u_idx != v_idx:
                        adj[u_idx, v_idx] = True
                        lat[u_idx, v_idx] = float(metrics.get("latency", 10.0))
        elif isinstance(topology, dict):
            for u, neighbors in topology.items():
                u_idx = int(u)
                if isinstance(neighbors, dict):
                    for v, metrics in neighbors.items():
                        v_idx = int(v)
                        adj[u_idx, v_idx] = True
                        if isinstance(metrics, dict):
                            lat[u_idx, v_idx] = float(metrics.get("latency", 10.0))
                        else:
                            lat[u_idx, v_idx] = float(metrics)
                elif isinstance(neighbors, (list, tuple, set)):
                    for v in neighbors:
                        v_idx = int(v)
                        adj[u_idx, v_idx] = True
                        lat[u_idx, v_idx] = 10.0

        return adj, lat

    def get_action_mask(self, node_id: Optional[int] = None) -> np.ndarray:
        """
        Get boolean action mask for valid neighbor selection from given node.

        Args:
            node_id: Target node index. Defaults to current_node.

        Returns:
            Boolean numpy array of size (num_nodes,) where True indicates a valid link.
        """
        if node_id is None:
            node_id = self.current_node
        if 0 <= node_id < self.num_nodes:
            return self.adj_matrix[node_id].copy()
        return np.zeros(self.num_nodes, dtype=bool)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment state conforming to Gymnasium v0.26+ API.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        self.step_count = 0
        self.dropped_packets = 0
        self.successful_deliveries = 0

        # Handle custom options if provided
        if options and "current_node" in options:
            self.current_node = int(options["current_node"])
        else:
            self.current_node = 0

        if options and "destination_node" in options:
            self.destination_node = int(options["destination_node"])
        else:
            self.destination_node = self.num_nodes - 1
            if self.current_node == self.destination_node and self.num_nodes > 1:
                self.destination_node = (self.current_node + 1) % self.num_nodes

        # Reset node queue lengths
        if options and "queue_depths" in options:
            self.queue_depths = np.array(options["queue_depths"], dtype=np.float32)
        else:
            self.queue_depths = self.np_random.uniform(
                low=0.0, high=self.max_queue_capacity * 0.2, size=self.num_nodes
            ).astype(np.float32)

        # Sync from data plane router nodes if passed in options
        if options and "router_nodes" in options:
            self.sync_from_data_plane(options["router_nodes"])

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one action step in the environment.

        Args:
            action: Selected target node index to route packet to.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self.step_count += 1
        action = int(action)

        # Validate action: check if link exists between current_node and target action node
        action_mask = self.get_action_mask(self.current_node)

        if action < 0 or action >= self.num_nodes or not action_mask[action]:
            # Invalid action execution: apply heavy penalty, increment drop counter, keep state unchanged
            self.dropped_packets += 1
            reward = -100.0
            terminated = False
            truncated = self.step_count >= self.max_steps
            info = self._get_info()
            info["error"] = "Invalid action: link does not exist"
            return self._get_observation(), reward, terminated, truncated, info

        # Valid action: forward packet to target neighbor
        prev_node = self.current_node
        self.current_node = action
        latency = float(self.link_latencies[prev_node, action])
        queue_length = float(self.queue_depths[self.current_node])

        # Check destination arrival
        if self.current_node == self.destination_node:
            self.successful_deliveries += 1
            reward = 50.0
            terminated = True
        # Check queue overflow at target node
        elif queue_length >= self.max_queue_capacity:
            self.dropped_packets += 1
            reward = -20.0
            terminated = False
        else:
            # Standard hop penalty proportional to latency and queue depth ratio
            reward = -(0.5 * latency + 1.0 * (queue_length / float(self.max_queue_capacity)))
            terminated = False

        truncated = (not terminated) and (self.step_count >= self.max_steps)
        obs = self._get_observation()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def sync_from_data_plane(self, router_nodes: Union[List[Any], Dict[str, Any]]) -> None:
        """
        Sync node queue depths from live data-plane RouterNode objects.
        """
        if isinstance(router_nodes, dict):
            for idx in range(self.num_nodes):
                key = str(idx)
                if key in router_nodes and hasattr(router_nodes[key], "queue_length"):
                    self.queue_depths[idx] = float(router_nodes[key].queue_length)
        elif isinstance(router_nodes, (list, tuple)):
            for idx, node in enumerate(router_nodes[: self.num_nodes]):
                if hasattr(node, "queue_length"):
                    self.queue_depths[idx] = float(node.queue_length)

    def render(self) -> Optional[str]:
        """
        Render environment visualization in ANSI or human mode.
        """
        output = (
            f"Node: {self.current_node} -> Target: {self.destination_node} | "
            f"Queues: {np.round(self.queue_depths, 1).tolist()} | "
            f"Drops: {self.dropped_packets} | Deliveries: {self.successful_deliveries}"
        )
        if self.render_mode == "ansi":
            return output
        elif self.render_mode == "human":
            print(output)
        return output

    def _get_observation(self) -> np.ndarray:
        """
        Construct normalized observation vector.
        """
        norm_queues = (self.queue_depths / float(self.max_queue_capacity)).astype(np.float32)
        dest = np.array([float(self.destination_node)], dtype=np.float32)
        adj_latencies = self.link_latencies[self.current_node].astype(np.float32)

        return np.concatenate([norm_queues, dest, adj_latencies]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """
        Construct diagnostic dictionary.
        """
        return {
            "current_node": self.current_node,
            "destination_node": self.destination_node,
            "dropped_packets": self.dropped_packets,
            "successful_deliveries": self.successful_deliveries,
            "step_count": self.step_count,
            "action_mask": self.get_action_mask(self.current_node),
        }

    def close(self) -> None:
        """Clean up resources."""
        pass