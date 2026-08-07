"""
tests/test_plane.py

Unit tests for router/plane.py: Packet construction, serialization,
validity constraints, and the BaseRouterNode interface.

Run with:  python -m pytest tests/test_plane.py -v
"""

import asyncio
import sys
import time
import unittest
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.plane import (  # noqa: E402
    BaseRouterNode,
    Packet,
    PacketValidationError,
    Priority,
    RouteEntry,
)


def make_packet(**overrides) -> Packet:
    defaults = dict(
        packet_id="",
        source="node-A",
        destination="node-B",
        payload=b"hello-world",
        priority=int(Priority.NORMAL),
        ttl=8,
    )
    defaults.update(overrides)
    return Packet(**defaults)


# --------------------------------------------------------------------------
# Packet construction
# --------------------------------------------------------------------------

class TestPacketConstruction(unittest.TestCase):

    def test_basic_construction(self):
        p = make_packet()
        self.assertEqual(p.source, "node-A")
        self.assertEqual(p.destination, "node-B")
        self.assertEqual(p.payload, b"hello-world")
        self.assertEqual(p.priority, int(Priority.NORMAL))
        self.assertEqual(p.ttl, 8)
        self.assertEqual(p.hop_history, [])
        self.assertTrue(p.packet_id)  # auto-generated uuid

    def test_packet_id_auto_generated_when_empty(self):
        p1 = make_packet(packet_id="")
        p2 = make_packet(packet_id="")
        self.assertNotEqual(p1.packet_id, p2.packet_id)

    def test_explicit_packet_id_preserved(self):
        p = make_packet(packet_id="fixed-id-123")
        self.assertEqual(p.packet_id, "fixed-id-123")

    def test_creation_time_defaults_to_now(self):
        before = time.time()
        p = make_packet()
        after = time.time()
        self.assertTrue(before <= p.creation_time <= after)

    def test_str_payload_is_utf8_encoded(self):
        p = make_packet(payload="hello")
        self.assertIsInstance(p.payload, bytes)
        self.assertEqual(p.payload, b"hello")

    def test_bytearray_payload_normalized_to_bytes(self):
        p = make_packet(payload=bytearray(b"abc"))
        self.assertIsInstance(p.payload, bytes)
        self.assertEqual(p.payload, b"abc")

    def test_default_priority_and_ttl(self):
        p = Packet(packet_id="", source="A", destination="B", payload=b"x")
        self.assertEqual(p.priority, int(Priority.NORMAL))
        self.assertEqual(p.ttl, 64)


# --------------------------------------------------------------------------
# Validity constraints
# --------------------------------------------------------------------------

class TestPacketValidation(unittest.TestCase):

    def test_invalid_payload_type_rejected(self):
        with self.assertRaises(PacketValidationError):
            make_packet(payload=12345)

    def test_empty_source_rejected(self):
        with self.assertRaises(PacketValidationError):
            make_packet(source="")

    def test_empty_destination_rejected(self):
        with self.assertRaises(PacketValidationError):
            make_packet(destination="")

    def test_source_equals_destination_rejected(self):
        with self.assertRaises(PacketValidationError):
            make_packet(source="same-node", destination="same-node")

    def test_priority_out_of_range_rejected(self):
        with self.assertRaises(PacketValidationError):
            make_packet(priority=99)
        with self.assertRaises(PacketValidationError):
            make_packet(priority=-1)

    def test_non_integer_priority_rejected(self):
        with self.assertRaises(PacketValidationError):
            make_packet(priority=1.5)

    def test_zero_or_negative_ttl_rejected(self):
        with self.assertRaises(PacketValidationError):
            make_packet(ttl=0)
        with self.assertRaises(PacketValidationError):
            make_packet(ttl=-5)

    def test_negative_creation_time_rejected(self):
        with self.assertRaises(PacketValidationError):
            make_packet(creation_time=-1.0)

    def test_non_list_hop_history_rejected(self):
        with self.assertRaises(PacketValidationError):
            make_packet(hop_history="not-a-list")

    def test_hop_history_with_non_string_entries_rejected(self):
        with self.assertRaises(PacketValidationError):
            make_packet(hop_history=["ok", 123])

    def test_record_hop_appends_and_decrements_ttl(self):
        p = make_packet(ttl=3)
        p2 = p.record_hop("router-1")
        self.assertEqual(p2.ttl, 2)
        self.assertEqual(p2.hop_history, ["router-1"])
        # original packet is untouched (immutable-update pattern)
        self.assertEqual(p.ttl, 3)
        self.assertEqual(p.hop_history, [])

    def test_record_hop_raises_when_ttl_would_exhaust(self):
        p = make_packet(ttl=1)
        with self.assertRaises(PacketValidationError):
            p.record_hop("router-1")

    def test_is_expired(self):
        p = make_packet(ttl=1)
        self.assertFalse(p.is_expired())

    def test_age_seconds(self):
        p = make_packet(creation_time=time.time() - 10)
        self.assertGreaterEqual(p.age_seconds(), 10)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

class TestPacketSerialization(unittest.TestCase):

    def test_to_dict_hex_encodes_payload(self):
        p = make_packet(payload=b"\x00\x01\xff")
        d = p.to_dict()
        self.assertEqual(d["payload"], "0001ff")

    def test_json_round_trip(self):
        p = make_packet(hop_history=["h1", "h2"])
        raw = p.to_json()
        p2 = Packet.from_json(raw)
        self.assertEqual(p, p2)

    def test_dict_round_trip(self):
        p = make_packet()
        p2 = Packet.from_dict(p.to_dict())
        self.assertEqual(p, p2)

    def test_bytes_round_trip(self):
        p = make_packet(payload=b"binary\x00payload\xff")
        raw = p.to_bytes()
        p2 = Packet.from_bytes(raw)
        self.assertEqual(p, p2)

    def test_bytes_round_trip_with_trailing_garbage_ignored(self):
        p = make_packet()
        raw = p.to_bytes() + b"\x00\x00\x00extra-noise"
        p2 = Packet.from_bytes(raw)
        self.assertEqual(p, p2)

    def test_from_bytes_too_short_raises(self):
        with self.assertRaises(PacketValidationError):
            Packet.from_bytes(b"\x00\x01")

    def test_from_bytes_truncated_body_raises(self):
        p = make_packet()
        raw = p.to_bytes()
        with self.assertRaises(PacketValidationError):
            Packet.from_bytes(raw[:-5])

    def test_from_json_invalid_json_raises(self):
        with self.assertRaises(PacketValidationError):
            Packet.from_json("{not valid json")

    def test_from_dict_missing_payload_raises(self):
        d = make_packet().to_dict()
        del d["payload"]
        with self.assertRaises(PacketValidationError):
            Packet.from_dict(d)

    def test_from_dict_bad_hex_payload_raises(self):
        d = make_packet().to_dict()
        d["payload"] = "not-hex!!"
        with self.assertRaises(PacketValidationError):
            Packet.from_dict(d)


# --------------------------------------------------------------------------
# BaseRouterNode — exercised through a minimal concrete subclass
# --------------------------------------------------------------------------

class LinearRouterNode(BaseRouterNode):
    """
    Minimal concrete BaseRouterNode used only for testing: routes by exact
    destination match and "forwards" by handing the packet to a peer node's
    queue directly (simulating a link between two in-process nodes).
    """

    def __init__(self, node_id: str, buffer_size: int = 4):
        super().__init__(node_id, buffer_size)
        self.peers: dict = {}
        self.delivered: list = []

    def link_to(self, peer: "LinearRouterNode") -> None:
        self.peers[peer.node_id] = peer
        self.add_route(peer.node_id, peer.node_id, metric=1)

    def lookup_route(self, destination: str) -> Optional[RouteEntry]:
        return self._routing_table.get(destination)

    async def forward(self, packet: Packet) -> Tuple[bool, Optional[str]]:
        if packet.destination == self.node_id:
            self.delivered.append(packet)
            return True, None
        route = self.lookup_route(packet.destination)
        if route is None:
            return False, None
        peer = self.peers.get(route.next_hop)
        if peer is None:
            return False, None
        accepted = await peer.enqueue(packet)
        return accepted, route.next_hop if accepted else None


class TestBaseRouterNode(unittest.TestCase):

    def test_cannot_instantiate_abstract_base(self):
        with self.assertRaises(TypeError):
            BaseRouterNode("bad-node")  # type: ignore[abstract]

    def test_construction_validates_args(self):
        with self.assertRaises(ValueError):
            LinearRouterNode("", buffer_size=4)
        with self.assertRaises(ValueError):
            LinearRouterNode("node-1", buffer_size=0)

    def test_initial_state(self):
        node = LinearRouterNode("node-1", buffer_size=4)
        self.assertEqual(node.buffer_size, 4)
        self.assertEqual(node.queue_length, 0)
        self.assertFalse(node.is_full)
        self.assertTrue(node.is_empty)
        self.assertEqual(node.routing_table, {})

    def test_add_and_remove_route(self):
        node = LinearRouterNode("node-1")
        node.add_route("node-2", "node-2", metric=3)
        self.assertIn("node-2", node.routing_table)
        self.assertEqual(node.routing_table["node-2"].metric, 3)
        node.remove_route("node-2")
        self.assertNotIn("node-2", node.routing_table)

    def test_enqueue_dequeue(self):
        async def run():
            node = LinearRouterNode("node-1", buffer_size=2)
            p = make_packet(source="node-0", destination="node-1")
            ok = await node.enqueue(p)
            self.assertTrue(ok)
            self.assertEqual(node.queue_length, 1)
            out = await node.dequeue()
            self.assertEqual(out.packet_id, p.packet_id)
            self.assertEqual(node.queue_length, 0)

        asyncio.run(run())

    def test_enqueue_rejected_when_buffer_full(self):
        async def run():
            node = LinearRouterNode("node-1", buffer_size=1)
            p1 = make_packet(source="node-0", destination="node-1")
            p2 = make_packet(source="node-0", destination="node-1")
            self.assertTrue(await node.enqueue(p1))
            self.assertTrue(node.is_full)
            self.assertFalse(await node.enqueue(p2))

        asyncio.run(run())

    def test_forward_delivers_locally(self):
        async def run():
            node = LinearRouterNode("node-B")
            p = make_packet(source="node-A", destination="node-B")
            success, next_hop = await node.forward(p)
            self.assertTrue(success)
            self.assertIsNone(next_hop)
            self.assertEqual(len(node.delivered), 1)

        asyncio.run(run())

    def test_forward_routes_to_linked_peer(self):
        async def run():
            a = LinearRouterNode("node-A")
            b = LinearRouterNode("node-B")
            a.link_to(b)
            p = make_packet(source="node-A", destination="node-B")
            success, next_hop = await a.forward(p)
            self.assertTrue(success)
            self.assertEqual(next_hop, "node-B")
            self.assertEqual(b.queue_length, 1)

        asyncio.run(run())

    def test_forward_with_no_route_fails(self):
        async def run():
            a = LinearRouterNode("node-A")
            p = make_packet(source="node-A", destination="node-Z")
            success, next_hop = await a.forward(p)
            self.assertFalse(success)
            self.assertIsNone(next_hop)

        asyncio.run(run())

    def test_process_one_end_to_end(self):
        async def run():
            a = LinearRouterNode("node-A")
            b = LinearRouterNode("node-B")
            a.link_to(b)
            p = make_packet(source="node-A", destination="node-B", ttl=5)
            await a.enqueue(p)
            result = await a.process_one()
            self.assertIsNotNone(result)
            self.assertEqual(result.ttl, 4)
            self.assertEqual(result.hop_history, ["node-A"])
            self.assertEqual(b.queue_length, 1)

        asyncio.run(run())

    def test_process_one_drops_expired_packet(self):
        async def run():
            a = LinearRouterNode("node-A")
            p = make_packet(source="node-A", destination="node-B", ttl=1)
            # Manually simulate an already-exhausted ttl by constructing
            # with the minimum valid ttl then checking drop-on-expiry path
            # via a packet whose ttl we force to 0 post-construction.
            object.__setattr__(p, "ttl", 0)
            await a.enqueue(p)
            result = await a.process_one()
            self.assertIsNone(result)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)