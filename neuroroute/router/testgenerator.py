"""
Unit tests for TrafficGenerator and TrafficStream in router/generator.py.
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neuroroute.router.generator import TrafficGenerator, TrafficStream
from neuroroute.router.plane import Priority, RouterNode


class TestTrafficGenerator(unittest.TestCase):

    def test_stream_construction_validation(self):
        """Test validation rules for TrafficStream initialization."""
        # Valid stream
        stream = TrafficStream(stream_id="s1", source="node-A", destination="node-B", rate_pps=50.0)
        self.assertEqual(stream.stream_id, "s1")
        self.assertEqual(stream.source, "node-A")
        self.assertEqual(stream.destination, "node-B")
        self.assertEqual(stream.rate_pps, 50.0)

        # Auto-generated stream_id
        s_auto = TrafficStream(stream_id="", source="node-A", destination="node-B", rate_pps=10.0)
        self.assertTrue(s_auto.stream_id.startswith("stream-"))

        # Invalid rate_pps <= 0
        with self.assertRaises(ValueError):
            TrafficStream(stream_id="bad", source="A", destination="B", rate_pps=0.0)

        with self.assertRaises(ValueError):
            TrafficStream(stream_id="bad", source="A", destination="B", rate_pps=-5.0)

        # Same source and destination
        with self.assertRaises(ValueError):
            TrafficStream(stream_id="bad", source="same", destination="same", rate_pps=10.0)

    def test_add_remove_and_update_rate(self):
        """Test adding streams, updating rates dynamically, and removing streams."""
        gen = TrafficGenerator()
        s_id = gen.create_stream(source="node-A", destination="node-B", rate_pps=20.0)

        self.assertIn(s_id, gen.streams)
        self.assertEqual(gen.streams[s_id].rate_pps, 20.0)

        # Dynamic rate tuning
        gen.update_rate(s_id, 100.0)
        self.assertEqual(gen.streams[s_id].rate_pps, 100.0)

        # Invalid rate update
        with self.assertRaises(ValueError):
            gen.update_rate(s_id, -10.0)

        # Remove stream
        gen.remove_stream(s_id)
        self.assertNotIn(s_id, gen.streams)

    def test_burst_generation(self):
        """Test high-priority congestion burst flow generation."""
        async def run():
            node_a = RouterNode("node-A", buffer_size=100)
            gen = TrafficGenerator(target_nodes={"node-A": node_a})

            burst_packets = await gen.trigger_burst(
                source="node-A",
                destination="node-B",
                count=15,
                priority=int(Priority.CRITICAL),
                payload_size=128,
            )

            self.assertEqual(len(burst_packets), 15)
            self.assertEqual(gen.burst_count, 15)
            self.assertEqual(gen.generated_count, 15)

            # Packets should be enqueued into node_a
            self.assertEqual(node_a.queue_length, 15)

            # Check priority of generated burst packets
            for pkt in burst_packets:
                self.assertEqual(pkt.priority, int(Priority.CRITICAL))
                self.assertEqual(pkt.source, "node-A")
                self.assertEqual(pkt.destination, "node-B")
                self.assertEqual(len(pkt.payload), 128)

        asyncio.run(run())

    def test_poisson_arrival_rates(self):
        """Test Poisson stream execution and empirical packet generation rate."""
        async def run():
            gen = TrafficGenerator(seed=42)

            # High rate stream (100 packets per second -> ~50 packets in 0.5s)
            gen.create_stream(source="node-A", destination="node-B", rate_pps=100.0)

            gen.start()
            await asyncio.sleep(0.4)
            await gen.stop()

            # Expect approximately 40 ± 20 packets given Poisson randomness over 0.4s
            stats = gen.get_stats()
            self.assertGreater(stats["total_generated"], 15)
            self.assertLess(stats["total_generated"], 80)

        asyncio.run(run())

    def test_generator_with_router_nodes_integration(self):
        """Test integration of TrafficGenerator with active RouterNodes."""
        async def run():
            node_a = RouterNode("node-A", buffer_size=200)
            node_b = RouterNode("node-B", buffer_size=200)
            node_a.link_peer(node_b)

            gen = TrafficGenerator(target_nodes={"node-A": node_a}, seed=123)
            gen.create_stream(source="node-A", destination="node-B", rate_pps=80.0)

            node_a.start()
            node_b.start()
            gen.start()

            await asyncio.sleep(0.3)

            await gen.stop()
            await node_a.stop()
            await node_b.stop()

            # Packets generated at node A should be processed/delivered to node B
            self.assertGreater(len(node_b.delivered), 5)
            self.assertGreaterEqual(node_a.processed_count, 5)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
