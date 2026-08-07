

from __future__ import annotations

import abc
import asyncio
import json
import struct
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


#Errors 

class PacketValidationError(ValueError):
    """Raised when a Packet fails construction or validity constraints."""



class Priority(IntEnum):
    """Standard packet priority levels used across the data plane."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# --------------------------------------------------------------------------
# Packet
# --------------------------------------------------------------------------

@dataclass
class Packet:
    """
    Canonical unit of data transferred between router nodes.

    Fields
    ------
    packet_id:      Unique identifier for this packet. Auto-generated
                     (uuid4) if left empty.
    source:         Identifier of the originating node.
    destination:    Identifier of the target node.
    payload:        Raw payload bytes (str payloads are UTF-8 encoded
                     automatically).
    priority:       Integer priority, see `Priority` (0-3).
    ttl:            Time-to-live, decremented on every hop. Must stay > 0.
    creation_time:  Unix timestamp of creation (defaults to time.time()).
    hop_history:    Ordered list of node_ids the packet has traversed.
    """

    packet_id: str
    source: str
    destination: str
    payload: bytes
    priority: int = int(Priority.NORMAL)
    ttl: int = 64
    creation_time: float = field(default_factory=time.time)
    hop_history: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.packet_id:
            self.packet_id = str(uuid.uuid4())

        if isinstance(self.payload, str):
            self.payload = self.payload.encode("utf-8")   # so that regular text can be used as payload

        if not isinstance(self.payload, (bytes, bytearray)):  # verify the data in payload is 
            raise PacketValidationError(
                f"payload must be bytes or str, got {type(self.payload)!r}"
            )

        self.payload = bytes(self.payload) # ensure payload is immutable bytes

        if not isinstance(self.source, str) or not self.source: # verify source is a non-empty string
            raise PacketValidationError("source must be a non-empty string")


        if not isinstance(self.destination, str) or not self.destination:  # verify destination is a non-empty string
            raise PacketValidationError("destination must be a non-empty string")

        if self.source == self.destination: # verify source and destination are not the same
            raise PacketValidationError("source and destination must differ")

        if not isinstance(self.priority, int) or not (0 <= self.priority <= 3): # verify priority is an integer within the range [0, 3]
            raise PacketValidationError(
                "priority must be an int within [0, 3] (see Priority enum)"
            )
        if not isinstance(self.ttl, int) or self.ttl <= 0: # verify ttl is a positive integer
            raise PacketValidationError("ttl must be a positive integer")

        if self.creation_time < 0: # verify creation_time is non-negative
            raise PacketValidationError("creation_time must be non-negative")

        if not isinstance(self.hop_history, list) or not all(
            isinstance(h, str) for h in self.hop_history
        ): # verify hop_history is a list of strings
            raise PacketValidationError("hop_history must be a list of node-id strings")

    # ---- lifecycle helpers -------------------------------------------------

    def record_hop(self, node_id: str) -> "Packet":
        """
        Return a *new* Packet reflecting transit through `node_id`:
        ttl decremented by one and node_id appended to hop_history.

        Immutable-update style keeps packets safe to share across
        concurrently-running coroutines/threads without aliasing bugs.
        """
        if not node_id:
            raise PacketValidationError("node_id must be a non-empty string")
        if self.ttl - 1 <= 0:
            raise PacketValidationError(f"packet {self.packet_id} expired (ttl exhausted)")
        return Packet(
            packet_id=self.packet_id,
            source=self.source,
            destination=self.destination,
            payload=self.payload,
            priority=self.priority,
            ttl=self.ttl - 1,
            creation_time=self.creation_time,
            hop_history=[*self.hop_history, node_id],
        )

    def is_expired(self) -> bool:
        return self.ttl <= 0

    def age_seconds(self, now: Optional[float] = None) -> float:
        return (now if now is not None else time.time()) - self.creation_time

    # ---- serialization -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict representation (payload hex-encoded)."""
        d = asdict(self)
        d["payload"] = self.payload.hex()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Packet":
        data = dict(data)
        try:
            payload_hex = data.pop("payload")
        except KeyError as exc:
            raise PacketValidationError("serialized packet missing 'payload' field") from exc
        try:
            payload = bytes.fromhex(payload_hex)
        except (TypeError, ValueError) as exc:
            raise PacketValidationError("payload field is not valid hex") from exc
        try:
            return cls(payload=payload, **data)
        except TypeError as exc:
            raise PacketValidationError(f"malformed packet dict: {exc}") from exc

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "Packet":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PacketValidationError(f"invalid packet JSON: {exc}") from exc
        return cls.from_dict(data)

    def to_bytes(self) -> bytes:
        """
        Wire format: 4-byte big-endian length prefix followed by UTF-8 JSON.
        Length-prefixing keeps this compatible with streaming transports
        where message boundaries aren't otherwise preserved.
        """
        body = self.to_json().encode("utf-8")
        header = struct.pack(">I", len(body))
        return header + body

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Packet":
        if len(raw) < 4:
            raise PacketValidationError("byte stream too short to contain a valid packet header")
        (length,) = struct.unpack(">I", raw[:4])
        body = raw[4:4 + length]
        if len(body) != length:
            raise PacketValidationError("byte stream length does not match declared packet length")
        return cls.from_json(body.decode("utf-8"))

    def __repr__(self) -> str:
        return (
            f"Packet(id={self.packet_id[:8]}…, {self.source}->{self.destination}, "
            f"priority={self.priority}, ttl={self.ttl}, hops={len(self.hop_history)})"
        )


# --------------------------------------------------------------------------
# Routing table entries
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteEntry:
    """A single routing table entry: where to send matching packets next."""
    destination_prefix: str
    next_hop: str
    metric: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# BaseRouterNode
# --------------------------------------------------------------------------

class BaseRouterNode(abc.ABC):
    """
    Abstract interface every router node in the data plane must implement.

    Concerns owned by this base class:
      * Bounded inbound buffer (`asyncio.Queue`), exposing queue length and
        buffer size for backpressure/telemetry.
      * A routing table mapping destination prefixes -> `RouteEntry`.
      * The `enqueue`/`dequeue` primitives used by the simulation scheduler.

    Concerns left to subclasses:
      * `lookup_route`   — how a destination is resolved (exact match,
                            longest-prefix match, hashing, etc.).
      * `forward`         — the actual forwarding decision/side effect for
                            a dequeued packet (e.g. handing it to a peer
                            node's queue, a NIC simulation, etc.).
    """

    def __init__(self, node_id: str, buffer_size: int = 1024) -> None:
        if not node_id:
            raise ValueError("node_id must be a non-empty string")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer")
        self.node_id = node_id
        self._buffer_size = buffer_size
        self._queue: "asyncio.Queue[Packet]" = asyncio.Queue(maxsize=buffer_size)
        self._routing_table: Dict[str, RouteEntry] = {}

    # ---- introspection -------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Maximum number of packets this node's inbound queue can hold."""
        return self._buffer_size

    @property
    def queue_length(self) -> int:
        """Current number of packets waiting in the inbound queue."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        return self._queue.full()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    @property
    def routing_table(self) -> Dict[str, RouteEntry]:
        """Read-only snapshot of the routing table."""
        return dict(self._routing_table)

    # ---- routing table management -------------------------------------

    def add_route(self, destination_prefix: str, next_hop: str, metric: int = 1) -> None:
        if not destination_prefix:
            raise ValueError("destination_prefix must be a non-empty string")
        if not next_hop:
            raise ValueError("next_hop must be a non-empty string")
        self._routing_table[destination_prefix] = RouteEntry(
            destination_prefix=destination_prefix, next_hop=next_hop, metric=metric
        )

    def remove_route(self, destination_prefix: str) -> None:
        self._routing_table.pop(destination_prefix, None)

    @abc.abstractmethod
    def lookup_route(self, destination: str) -> Optional[RouteEntry]:
        """Resolve a destination to a `RouteEntry`, or None if unreachable."""
        raise NotImplementedError

    # ---- queueing --------------------------------------------------------

    async def enqueue(self, packet: Packet) -> bool:
        """
        Attempt to place a packet on the inbound queue without blocking.
        Returns False (drop) if the buffer is full, True if accepted.
        """
        if self._queue.full():
            return False
        await self._queue.put(packet)
        return True

    async def dequeue(self) -> Packet:
        """Await and remove the next packet from the inbound queue."""
        return await self._queue.get()

    # ---- forwarding --------------------------------------------------------

    @abc.abstractmethod
    async def forward(self, packet: Packet) -> Tuple[bool, Optional[str]]:
        """
        Forward `packet` toward its destination.

        Returns
        -------
        (success, next_hop_node_id): `success` indicates whether the
        forwarding action was accepted (e.g. downstream queue had room);
        `next_hop_node_id` is the id of the node the packet was handed to,
        or None if the packet was dropped / had no route / was delivered
        locally.
        """
        raise NotImplementedError

    async def process_one(self) -> Optional[Packet]:
        """
        Convenience driver: dequeue one packet, drop it if expired,
        forward it, and return the hop-updated packet on success or
        None if it was dropped.
        """
        packet = await self.dequeue()
        if packet.is_expired():
            return None
        success, next_hop = await self.forward(packet)
        if not success or next_hop is None:
            return None
        return packet.record_hop(self.node_id)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(node_id={self.node_id!r}, "
            f"queue={self.queue_length}/{self.buffer_size}, "
            f"routes={len(self._routing_table)})"
        )