import asyncio
import pytest
from neuroroute.network.algorithms import DijkstraRouter
from neuroroute.network.chaos import ChaosScheduler
from neuroroute.network.topology import TopologyManager


@pytest.fixture
def triangle_topology():
  """Creates a 3-node triangle topology:

  A --[2]--> B --[2]--> C
  A --------[10]-------> C
  """
  topo = TopologyManager()
  topo.add_link("A", "B", latency=2.0)
  topo.add_link("B", "C", latency=2.0)
  topo.add_link("A", "C", latency=10.0)
  return topo


def test_link_failure_path_recalculation(triangle_topology):
  router = DijkstraRouter(triangle_topology)
  chaos = ChaosScheduler(triangle_topology)

  # Initial optimal path: A -> B -> C (latency = 4)
  assert router.get_shortest_path("A", "C") == ["A", "B", "C"]

  # Fail link B -> C
  chaos.fail_link("B", "C")

  # Recalculates to direct link: A -> C (latency = 10)
  assert router.get_shortest_path("A", "C") == ["A", "C"]
  assert router.get_next_hop("A", "C") == "C"


def test_link_restoration(triangle_topology):
  router = DijkstraRouter(triangle_topology)
  chaos = ChaosScheduler(triangle_topology)

  chaos.fail_link("B", "C")
  assert router.get_shortest_path("A", "C") == ["A", "C"]

  # Restore link B -> C
  chaos.restore_link("B", "C")
  assert router.get_shortest_path("A", "C") == ["A", "B", "C"]


def test_latency_spike_rerouting(triangle_topology):
  router = DijkstraRouter(triangle_topology)
  chaos = ChaosScheduler(triangle_topology)

  # Initial path: A -> B -> C (2 + 2 = 4)
  assert router.get_shortest_path("A", "C") == ["A", "B", "C"]

  # Spike latency on A -> B by 10x (2.0 * 10 = 20.0)
  chaos.spike_latency("A", "B", chaos_factor=10.0)

  # Path A -> B -> C total latency becomes 22, so router prefers direct A -> C (10)
  assert router.get_shortest_path("A", "C") == ["A", "C"]


def test_node_dropout(triangle_topology):
  router = DijkstraRouter(triangle_topology)
  chaos = ChaosScheduler(triangle_topology)

  # Drop node B completely
  chaos.node_dropout("B")

  assert triangle_topology.get_neighbours("B") == []
  assert router.get_shortest_path("A", "C") == ["A", "C"]

  # Restore node B
  chaos.node_restore("B")
  assert router.get_shortest_path("A", "C") == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_async_chaos_scheduler_loop(triangle_topology):
  chaos = ChaosScheduler(triangle_topology)

  # Run scheduler for 0.15s with 0.05s intervals
  asyncio.create_task(chaos.start(interval_seconds=0.05, duration_seconds=0.15))
  await asyncio.sleep(0.2)

  chaos.stop()
  assert chaos._running is False