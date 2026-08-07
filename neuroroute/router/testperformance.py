"""
neuroroute/router/testperformance.py

Performance validation tests benchmarking per-hop packet processing latency
(target: < 5 microseconds per hop) and comparing FastQueue vs standard asyncio.Queue.
"""

import asyncio
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neuroroute.router.plane import FastQueue, FastRouterNode, Packet, Priority, RouterNode


class TestPerformanceOptimization(unittest.TestCase):

    def setUp(self):
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

    def test_fast_queue_vs_asyncio_queue_throughput(self):
        """Compare FastQueue put/get latency against standard asyncio.Queue."""
        async def run():
            num_ops = 50000

            # 1. Standard asyncio.Queue
            std_queue = asyncio.Queue(maxsize=num_ops)
            t0 = time.perf_counter()
            for i in range(num_ops):
                await std_queue.put(i)
            for i in range(num_ops):
                await std_queue.get()
            t1 = time.perf_counter()
            std_time = t1 - t0

            # 2. FastQueue
            fast_queue = FastQueue(maxsize=num_ops)
            t2 = time.perf_counter()
            for i in range(num_ops):
                fast_queue.put_nowait(i)
            for i in range(num_ops):
                fast_queue.get_nowait()
            t3 = time.perf_counter()
            fast_time = t3 - t2

            print(f"\n[Queue Benchmark] Standard asyncio.Queue: {std_time:.4f}s | FastQueue: {fast_time:.4f}s")
            self.assertLess(fast_time, std_time)

        asyncio.run(run())

    def test_route_lookup_cache_hit(self):
        """Verify route lookup cache provides O(1) speedup."""
        node = RouterNode("node-A")
        # Add 50 prefix routes
        for i in range(50):
            node.add_route(f"192.168.{i}.", f"gateway-{i}")

        node.add_route("10.0.0.1", "target-node")

        # Warm up cache
        node.lookup_route("10.0.0.1")

        # Measure 100,000 cached lookups
        t0 = time.perf_counter()
        for _ in range(100000):
            res = node.lookup_route("10.0.0.1")
        t1 = time.perf_counter()

        elapsed = t1 - t0
        per_lookup_ns = (elapsed / 100000) * 1e9
        print(f"\n[Route Cache Benchmark] 100,000 lookups in {elapsed:.4f}s ({per_lookup_ns:.2f} ns/lookup)")
        self.assertEqual(res.next_hop, "target-node")
        self.assertLess(per_lookup_ns, 1000.0)  # Sub-microsecond lookup

    def test_per_hop_processing_latency_under_5_microseconds(self):
        """Benchmark packet processing latency targeting < 5 µs per hop."""
        num_packets = 10000
        hops = 4

        nodes = [FastRouterNode(f"node-{i}", buffer_size=50000) for i in range(hops + 1)]
        for i in range(hops):
            nodes[i].link_peer(nodes[i + 1])
            nodes[i].add_route(f"node-{hops}", f"node-{i + 1}")

        dest = f"node-{hops}"
        packets = [
            Packet(packet_id="", source="node-0", destination=dest, payload=b"test", priority=int(Priority.NORMAL), ttl=64)
            for _ in range(num_packets)
        ]

        # Enqueue into source node
        for p in packets:
            nodes[0].enqueue_fast(p)

        # Benchmark traversal
        t0 = time.perf_counter()
        while any(not n.is_empty for n in nodes):
            for node in nodes:
                node.process_one_fast()
        t1 = time.perf_counter()

        total_hops_processed = len(nodes[-1].delivered) * hops
        elapsed = t1 - t0
        per_hop_us = (elapsed / max(1, total_hops_processed)) * 1e6

        print(f"\n[Latency Benchmark] Processed {len(nodes[-1].delivered)} packets across {total_hops_processed} total hops")
        print(f"Elapsed: {elapsed:.4f}s | Average Per-Hop Processing Latency: {per_hop_us:.3f} µs")

        self.assertEqual(len(nodes[-1].delivered), num_packets)
        # Target acceptance criteria: < 5 microseconds per hop!
        self.assertLess(per_hop_us, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
