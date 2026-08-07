"""
neuroroute/router/profile_router.py

Profiling script analyzing RouterNode and FastRouterNode execution overhead
using cProfile under heavy traffic simulation.
"""

import asyncio
import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neuroroute.router.plane import FastRouterNode, Packet, Priority, RouterNode


def create_topology(node_cls, count: int = 10):
    nodes = [node_cls(f"node-{i}", buffer_size=50000) for i in range(count)]
    for i in range(count - 1):
        nodes[i].link_peer(nodes[i + 1])
        # Add multi-hop route to destination node-(count-1)
        nodes[i].add_route(f"node-{count - 1}", f"node-{i + 1}")
    return nodes


def profile_heavy_traffic(num_packets: int = 10000):
    print(f"--- Profiling RouterNode Fast-Path under {num_packets} packets ---")
    nodes = create_topology(FastRouterNode, count=5)
    source = nodes[0]
    dest = f"node-4"

    # Generate packets
    packets = [
        Packet(packet_id="", source="node-0", destination=dest, payload=b"test-payload", priority=int(Priority.NORMAL), ttl=64)
        for _ in range(num_packets)
    ]

    for p in packets:
        source.enqueue_fast(p)

    profiler = cProfile.Profile()
    profiler.enable()

    t0 = time.perf_counter()
    processed = 0
    # Traverse through 4 hops
    while any(not n.is_empty for n in nodes):
        for node in nodes:
            node.process_one_fast()
    t1 = time.perf_counter()

    profiler.disable()

    elapsed = t1 - t0
    total_hops = len(nodes[-1].delivered) * 4
    per_hop_us = (elapsed / max(1, total_hops)) * 1e6 if total_hops > 0 else 0.0

    print(f"Processed {len(nodes[-1].delivered)} delivered packets across {total_hops} total hops in {elapsed:.4f}s")
    print(f"Average Per-Hop Latency: {per_hop_us:.2f} µs")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    print("\n--- cProfile Top 10 Cumulative Time Functions ---")
    stats.print_stats(10)


if __name__ == "__main__":
    profile_heavy_traffic(10000)
