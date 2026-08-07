import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock
import pytest
from neuroroute.network.algorithms import Dijkstras
from neuroroute.network.topology import TopologyManager


@dataclass
class Packet:
    packet_id: str
    source: str
    destination: str
    current_node: str
    payload: str = "test_payload"


@pytest.fixture
def complex_topology(tmp_path):
    """Creates a 6-node complex mesh topology:

        (B) ──[1]── (D)
        ╱   ╲         │
    [10]   [2]      [1]
    ╱       ╲       │
    (A) ──[1]── (C) ──[2]── (E)

    Node F is completely disconnected (unreachable).
    """
    config_file = tmp_path / "complex_mesh.json"
    config_file.write_text("""{
            "nodes": ["A", "B", "C", "D", "E", "F"],
            "links": [
                {"from": "A", "to": "B", "latency": 10},
                {"from": "A", "to": "C", "latency": 1},
                {"from": "B", "to": "C", "latency": 2},
                {"from": "B", "to": "D", "latency": 1},
                {"from": "C", "to": "D", "latency": 2},
                {"from": "C", "to": "E", "latency": 2},
                {"from": "D", "to": "E", "latency": 1}
            ]
        }""")

    topo = TopologyManager()
    topo.load_topology(str(config_file))
    return topo


def test_dijkstra_complex_multi_hop(complex_topology):
    """Validates shortest path selection on a multi-path graph."""
    router = Dijkstras(complex_topology)

    # A -> C (1) -> E (2) = Total 3 (Shortest!)
    path = router.get_shortest_path("A", "E")
    assert path == ["A", "C", "E"]
    assert router.get_next_hop("A", "E") == "C"


def test_dijkstra_dynamic_reroute(complex_topology):
    """Verifies Dijkstra recalculates when link latency degrades."""
    router = Dijkstras(complex_topology)

    # Initial shortest path A -> C -> E (latency 3)
    assert router.get_shortest_path("A", "E") == ["A", "C", "E"]

    # Degrade link C -> E latency from 2 to 10
    complex_topology.update_link("C", "E", latency=10.0)

    # New shortest path should reroute: A -> C (1) -> D (2) -> E (1) = Total 4
    new_path = router.get_shortest_path("A", "E")
    assert new_path == ["A", "C", "D", "E"]
    assert router.get_next_hop("C", "E") == "D"


def test_dijkstra_unreachable_node(complex_topology):
    """Ensures unreachable node returns empty path and None next-hop."""
    router = Dijkstras(complex_topology)

    path = router.get_shortest_path("A", "F")
    assert path == []
    assert router.get_next_hop("A", "F") is None


def test_dijkstra_same_source_and_destination(complex_topology):
    """Ensures source == destination yields single node path."""
    router = Dijkstras(complex_topology)

    path = router.get_shortest_path("A", "A")
    assert path == ["A"]
    assert router.get_next_hop("A", "A") == "A"


@pytest.mark.asyncio
async def test_async_data_plane_forwarding_loop(tmp_path):
    """Tests asynchronous forwarding pipeline passing a packet through line topology: A -> B -> C."""
    config_file = tmp_path / "line_mesh.json"
    config_file.write_text("""{
            "nodes": ["A", "B", "C"],
            "links": [
                {"from": "A", "to": "B", "latency": 1},
                {"from": "B", "to": "C", "latency": 1}
            ]
        }""")

    topo = TopologyManager()
    topo.load_topology(str(config_file))
    router = Dijkstras(topo)

    # Simulate Async Node Queues
    node_queues = {
        "A": asyncio.Queue(),
        "B": asyncio.Queue(),
        "C": asyncio.Queue(),
    }

    async def forward_pipeline(packet: Packet) -> list[str]:
        route_taken = []
        while packet.current_node != packet.destination:
            curr = packet.current_node
            route_taken.append(curr)

            next_hop = router.get_next_hop(curr, packet.destination)
            assert (
                next_hop is not None
            ), f"No route found from {curr} to {packet.destination}"

            # Asynchronously enqueue packet to next hop
            await node_queues[next_hop].put(packet)

            # Dequeue packet at destination node
            packet = await asyncio.wait_for(node_queues[next_hop].get(), timeout=1.0)
            packet.current_node = next_hop

        route_taken.append(packet.current_node)
        return route_taken

    pkt = Packet(packet_id="pkt_001", source="A", destination="C", current_node="A")
    path_result = await forward_pipeline(pkt)

    assert path_result == ["A", "B", "C"]
    assert pkt.current_node == "C"