import json
from typing import Any, Optional


class TopologyManager:

  def __init__(self) -> None:
    self.graph: dict[str, dict[str, dict[str, Any]]] = {}

  def load_topology(self, filepath: str) -> None:
    """Loads network topology from a JSON file."""
    with open(filepath, "r") as f:
      data = json.load(f)

    self.graph.clear()
    for node in data.get("nodes", []):
      if node not in self.graph:
        self.graph[node] = {}

    for link in data.get("links", []):
      src = link["from"]
      dst = link["to"]
      latency = float(link.get("latency", 1.0))
      bandwidth = float(link.get("bandwidth", 100.0))
      self.add_link(src, dst, latency=latency, bandwidth=bandwidth)

  def add_link(
      self,
      start: str,
      end: str,
      latency: float = 1.0,
      bandwidth: float = 100.0,
      active: bool = True,
  ) -> None:
    """Adds or updates a directed link with active status tracking."""
    if start not in self.graph:
      self.graph[start] = {}
    if end not in self.graph:
      self.graph[end] = {}

    self.graph[start][end] = {
        "latency": latency,
        "base_latency": latency,
        "bandwidth": bandwidth,
        "active": active,
    }

  def update_link(
      self,
      start: str,
      end: str,
      latency: Optional[float] = None,
      bandwidth: Optional[float] = None,
      active: Optional[bool] = None,
  ) -> None:
    """Updates specific link metrics dynamically."""
    if start in self.graph and end in self.graph[start]:
      link = self.graph[start][end]
      if latency is not None:
        link["latency"] = latency
        link["base_latency"] = latency
      if bandwidth is not None:
        link["bandwidth"] = bandwidth
      if active is not None:
        link["active"] = active

  def set_link_status(self, start: str, end: str, active: bool) -> bool:
    """Enables or disables a specific link."""
    if start in self.graph and end in self.graph[start]:
      self.graph[start][end]["active"] = active
      return True
    return False

  def apply_latency_spike(
      self, start: str, end: str, chaos_factor: float
  ) -> bool:
    """Multiplies link latency by a chaos factor."""
    if start in self.graph and end in self.graph[start]:
      link = self.graph[start][end]
      link["latency"] = link["base_latency"] * chaos_factor
      return True
    return False

  def restore_link_latency(self, start: str, end: str) -> bool:
    """Restores link latency back to its base value."""
    if start in self.graph and end in self.graph[start]:
      link = self.graph[start][end]
      link["latency"] = link["base_latency"]
      return True
    return False

  def get_neighbours(self, node: str) -> list[str]:
    """Returns adjacent neighbor nodes connected via active links."""
    if node not in self.graph:
      return []
    return [
        neighbor
        for neighbor, metrics in self.graph[node].items()
        if metrics.get("active", True)
    ]

  def get_all_links(self) -> list[tuple[str, str]]:
    """Returns list of all directed link tuples (src, dst)."""
    links = []
    for src, neighbors in self.graph.items():
      for dst in neighbors:
        links.append((src, dst))
    return links