"""
neuroroute/router/generator.py

Packet generator module that simulates realistic, heavy network traffic flows
using Poisson distribution inter-arrival intervals and high-priority traffic burst spikes.
"""

import asyncio
import collections
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from neuroroute.router.plane import BaseRouterNode, Packet, Priority, PacketValidationError


@dataclass
class TrafficStream:
    """
    Configuration specification for a single traffic stream flow.

    Fields
    ------
    stream_id:     Unique stream identifier (auto-generated if empty).
    source:        Source node ID.
    destination:   Destination target node ID.
    rate_pps:      Average packet generation rate in packets per second (lambda > 0).
    payload_size:  Payload size in bytes for generated packets.
    priority:      Integer priority level (0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL).
    ttl:           Time-to-live value for generated packets.
    enabled:       Flag indicating if the stream is currently active.
    """

    stream_id: str
    source: str
    destination: str
    rate_pps: float
    payload_size: int = 64
    priority: int = int(Priority.NORMAL)
    ttl: int = 64
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.stream_id:
            self.stream_id = f"stream-{uuid.uuid4().hex[:8]}"
        if not self.source or not isinstance(self.source, str):
            raise ValueError("source must be a non-empty string")
        if not self.destination or not isinstance(self.destination, str):
            raise ValueError("destination must be a non-empty string")
        if self.source == self.destination:
            raise ValueError("source and destination must differ")
        if self.rate_pps <= 0:
            raise ValueError("rate_pps must be a positive float (> 0)")
        if self.payload_size < 0:
            raise ValueError("payload_size must be non-negative")


class TrafficGenerator:
    """
    Asynchronous traffic generator simulating realistic Poisson-distributed
    packet arrivals, dynamic rate tuning, and high-priority burst flows.
    """

    def __init__(
        self,
        target_nodes: Optional[Dict[str, BaseRouterNode]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the traffic generator.

        Args:
            target_nodes: Dictionary mapping node_id -> BaseRouterNode object.
            seed: Optional seed for reproducible random distribution sampling.
        """
        self.target_nodes: Dict[str, BaseRouterNode] = target_nodes if target_nodes is not None else {}
        self.streams: Dict[str, TrafficStream] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running: bool = False

        # Random number generator instance
        self.rng = random.Random(seed)

        # Telemetry metrics — bounded ring buffer keeps only the last 1000
        # packet references to prevent unbounded memory growth.
        self.generated_packets: collections.deque = collections.deque(maxlen=1000)
        self.generated_count: int = 0
        self.burst_count: int = 0

    # ---- Stream Management -------------------------------------------------

    def register_node(self, node: BaseRouterNode) -> None:
        """Register or update a target router node for packet injection."""
        self.target_nodes[node.node_id] = node

    def add_stream(self, stream: TrafficStream) -> str:
        """
        Add a traffic stream configuration.
        Returns the stream_id.
        """
        self.streams[stream.stream_id] = stream
        if self._running and stream.enabled:
            self._tasks[stream.stream_id] = asyncio.create_task(self._run_stream(stream))
        return stream.stream_id

    def create_stream(
        self,
        source: str,
        destination: str,
        rate_pps: float,
        payload_size: int = 64,
        priority: int = int(Priority.NORMAL),
        ttl: int = 64,
        stream_id: str = "",
    ) -> str:
        """Convenience method to construct and add a TrafficStream."""
        stream = TrafficStream(
            stream_id=stream_id,
            source=source,
            destination=destination,
            rate_pps=rate_pps,
            payload_size=payload_size,
            priority=priority,
            ttl=ttl,
        )
        return self.add_stream(stream)

    def remove_stream(self, stream_id: str) -> None:
        """Remove a traffic stream and cancel its async task if running."""
        stream = self.streams.pop(stream_id, None)
        task = self._tasks.pop(stream_id, None)
        if task and not task.done():
            task.cancel()

    def update_rate(self, stream_id: str, new_rate_pps: float) -> None:
        """
        Dynamically tune lambda (arrival rate) for a running stream.
        """
        if new_rate_pps <= 0:
            raise ValueError("new_rate_pps must be a positive float (> 0)")
        if stream_id in self.streams:
            self.streams[stream_id].rate_pps = float(new_rate_pps)

    # ---- Packet Generation & Bursts ----------------------------------------

    def _make_packet(self, stream: TrafficStream) -> Packet:
        """Construct a Packet instance from a stream specification."""
        payload = b"X" * stream.payload_size
        return Packet(
            packet_id="",
            source=stream.source,
            destination=stream.destination,
            payload=payload,
            priority=stream.priority,
            ttl=stream.ttl,
        )

    async def trigger_burst(
        self,
        source: str,
        destination: str,
        count: int = 10,
        priority: int = int(Priority.CRITICAL),
        payload_size: int = 64,
    ) -> List[Packet]:
        """
        Generate a sudden spike of high-priority packets simulating congestion bursts.

        Args:
            source: Source node ID.
            destination: Destination node ID.
            count: Number of burst packets to generate.
            priority: Priority level for burst packets (defaults to CRITICAL).
            payload_size: Payload size per burst packet.

        Returns:
            List of generated burst Packet objects.
        """
        if count <= 0:
            raise ValueError("burst count must be a positive integer")

        burst_packets: List[Packet] = []
        payload = b"B" * payload_size

        for _ in range(count):
            pkt = Packet(
                packet_id="",
                source=source,
                destination=destination,
                payload=payload,
                priority=priority,
                ttl=64,
            )
            burst_packets.append(pkt)
            self.generated_packets.append(pkt)
            self.generated_count += 1
            self.burst_count += 1

            # Inject into target node queue if registered
            if source in self.target_nodes:
                await self.target_nodes[source].enqueue(pkt)

        return burst_packets

    # ---- Worker Loop -------------------------------------------------------

    async def _run_stream(self, stream: TrafficStream) -> None:
        """
        Asynchronous worker loop modeling Poisson packet inter-arrivals.
        Inter-arrival delay is drawn from an exponential distribution:
        delay ~ Exp(lambda) where lambda = stream.rate_pps.
        """
        while self._running and stream.enabled:
            # Draw Poisson inter-arrival delay (exponentially distributed)
            # rate_pps is lambda (mean interval = 1 / lambda)
            delay = self.rng.expovariate(stream.rate_pps)
            
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break

            if not self._running or not stream.enabled:
                break

            packet = self._make_packet(stream)
            self.generated_packets.append(packet)
            self.generated_count += 1

            # Inject packet into source node's queue if registered
            target_node = self.target_nodes.get(stream.source)
            if target_node is not None:
                await target_node.enqueue(packet)

    # ---- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start async traffic generation loops for all registered streams."""
        if not self._running:
            self._running = True
            for stream_id, stream in self.streams.items():
                if stream.enabled and stream_id not in self._tasks:
                    self._tasks[stream_id] = asyncio.create_task(self._run_stream(stream))

    async def stop(self) -> None:
        """Stop all traffic generation tasks gracefully."""
        if self._running:
            self._running = False
            for task in list(self._tasks.values()):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            self._tasks.clear()

    # ---- Telemetry ---------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics dictionary."""
        return {
            "total_generated": self.generated_count,
            "burst_packets": self.burst_count,
            "active_streams": len(self.streams),
            "stream_rates": {s_id: s.rate_pps for s_id, s in self.streams.items()},
        }
