import heapq
import random
from typing import Optional
from neuroroute.network.topology import TopologyManager

class Dijkstras:
    def __init__(self, topo : TopologyManager) -> None:
        self.topo = topo

    def get_shortest_path(self, source : str, target : str) -> list[str]:
        if (source not in self.topo.graph or target not in self.topo.graph):
            if (source not in self.topo.graph):
                print(f"source not connected")
            elif (target not in self.topo.graph):
                print(f"target not connected")
            else:
                print(f"both source and target not connected")

            return []

        if source == target:        # should strip the inputs of left and right whitespaces
            return [source]

        # if no base case, then implement dijkstras
        distances: dict[str, float] = {source: 0.0}
        previous: dict[str, Optional[str]] = {source: None}     # what is this line doing
        pq: list[tuple[float, str]] = [(0.0, source)]
        visited: set[str] = set()

        while pq:       # pq is not empty
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited: continue
            visited.add(current_node)
            if current_node == target:
                break

            neighbours = self.topo.graph.get(current_node, {})
            for neighbour, metric in neighbours.items():
                latency = metric.get("latency", float("inf"))
                newdist = latency + current_dist

                if newdist < distances.get(neighbour, float("inf")):
                    distances[neighbour] = newdist
                    previous[neighbour] = current_node
                    heapq.heappush(pq, (newdist, neighbour))

        path: list[str] = []
        curr: Optional[str] = target
        while curr is not None:
            path.append(curr)
            curr = previous.get(curr)

        path.reverse()
        return path if (path and path[0] == source) else []

    def get_next_hop(self, current: str, destination: str) -> Optional[str]:
        if current == destination:
            return current

        path = self.get_shortest_path(current, destination)
        return path[1] if len(path) > 1 else None


class RoundRobin():
    def __init__(self, topo : TopologyManager) -> None:
        self.topo = topo
        self._indices: dict[str, int] = {}

    def get_next_hop(self, current: str, destination: str) -> Optional[str]:
        if current == destination:
            return current

        neighbours = self.topo.get_neighbours(current)
        if not neighbours:
            return None

        # Track state per node
        idx = self._indices.get(current, 0)
        next_hop = neighbours[idx % len(neighbours)]
        self._indices[current] = idx + 1
        return next_hop

class Random:
    def __init__(self, topo: TopologyManager) -> None:
        self.topo = topo

    def get_next_hop(self, current: str, destination: str) -> Optional[str]:
        if current == destination:
            return current

        neighbors = self.topo.get_neighbours(current)
        if not neighbors:
            return None

        return random.choice(neighbors)

    

