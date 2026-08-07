import pytest
from neuroroute.network.topology import TopologyManager
from neuroroute.network.algorithms import Dijkstras, RoundRobin, Random

@pytest.fixture
def sample_topo():
    topo = TopologyManager()
    topo.load_topology("configs/test-topology.json")
    # topo.print_graph()        # captured stdout will also be displayed
    return topo

def test_dijkstras(sample_topo):
    router = Dijkstras(sample_topo)

    path = router.get_shortest_path("nodeA", "nodeD")
    assert path == ["nodeA", "nodeB", "nodeD"]

    next_hop = router.get_next_hop("nodeA", "nodeD")
    assert next_hop == "nodeB"

def test_round_robin(sample_topo):
    router = RoundRobin(sample_topo)

    first_hop = router.get_next_hop("nodeA", "nodeD")
    second_hop = router.get_next_hop("nodeA", "nodeD")
    # third_hop = router.get_next_hop("nodeA", "nodeD")

    assert first_hop == "nodeB"
    assert second_hop == "nodeD"

def test_random_router_valid_neighbour(sample_topo):
    router = Random(sample_topo)
    next_hop = router.get_next_hop("nodeA", "nodeD")

    assert next_hop in ["nodeB", "nodeC"]

