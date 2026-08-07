import json
import pytest
from neuroroute.network.topology import TopologyManager


@pytest.fixture
def valid_topology_file(tmp_path):
    """Generates a valid 3-node topology JSON file."""
    file_path = tmp_path / "valid_mesh.json"
    data = {
        "nodes": ["nodeA", "nodeB", "nodeC"],
        "links": [
            {"from": "nodeA", "to": "nodeB", "latency": 5.0, "bandwidth": 100.0},
            {"from": "nodeB", "to": "nodeC", "latency": 2.0, "bandwidth": 50.0},
        ],
    }
    file_path.write_text(json.dumps(data))
    return str(file_path)


@pytest.fixture
def invalid_json_file(tmp_path):
    """Generates a malformed JSON file."""
    file_path = tmp_path / "invalid.json"
    file_path.write_text("{ malformed_json: true, ")
    return str(file_path)


@pytest.fixture
def missing_fields_file(tmp_path):
    """Generates a topology file missing critical fields."""
    file_path = tmp_path / "missing_fields.json"
    data = {
        "nodes": ["nodeA", "nodeB"],
        "links": [
            {"from": "nodeA"}  # Missing 'to' and 'latency'
        ],
    }
    file_path.write_text(json.dumps(data))
    return str(file_path)


def test_load_valid_topology(valid_topology_file):
    """Verifies correct loading of valid topology JSON."""
    topo = TopologyManager()
    topo.load_topology(valid_topology_file)

    assert "nodeA" in topo.graph
    assert "nodeB" in topo.graph
    assert "nodeC" in topo.graph
    assert topo.graph["nodeA"]["nodeB"]["latency"] == 5.0
    assert topo.get_neighbours("nodeA") == ["nodeB"]


def test_load_invalid_json_raises_error(invalid_json_file):
    """Ensures malformed JSON raises JSONDecodeError."""
    topo = TopologyManager()
    with pytest.raises(json.JSONDecodeError):
        topo.load_topology(invalid_json_file)


def test_load_missing_fields_raises_error(missing_fields_file):
    """Ensures missing link keys trigger exception handling."""
    topo = TopologyManager()
    with pytest.raises((KeyError, ValueError, Exception)):
        topo.load_topology(missing_fields_file)


def test_add_and_update_link(tmp_path):
    """Tests loading multiple links and mutating existing link metrics independently."""
    config_file = tmp_path / "multi_link.json"
    config_file.write_text(json.dumps({
        "nodes": ["A", "B", "C"],
        "links": [
            {"from": "A", "to": "B", "latency": 10.0, "bandwidth": 100.0},
            {"from": "A", "to": "C", "latency": 15.0, "bandwidth": 50.0},
        ],
    }))
    topo = TopologyManager()
    topo.load_topology(str(config_file))

    # Ensure both neighbors exist
    assert set(topo.get_neighbours("A")) == {"B", "C"}

    # Partial update: modify latency only
    topo.update_link("A", "B", latency=2.0)
    assert topo.graph["A"]["B"]["latency"] == 2.0
    assert topo.graph["A"]["B"]["bandwidth"] == 100.0  # Bandwidth untouched


def test_edge_cases_empty_and_disconnected(tmp_path):
    """Tests queries on non-existent nodes and disconnected components."""
    topo = TopologyManager()

    # Non-existent node query on empty graph
    assert topo.get_neighbours("non_existent_node") == []

    # Disconnected components loaded from topology file
    config_file = tmp_path / "disconnected.json"
    config_file.write_text(json.dumps({
        "nodes": ["A", "B", "X", "Y"],
        "links": [
            {"from": "A", "to": "B", "latency": 1.0},
            {"from": "X", "to": "Y", "latency": 1.0},
        ],
    }))
    topo.load_topology(str(config_file))

    assert topo.get_neighbours("A") == ["B"]
    assert topo.get_neighbours("X") == ["Y"]
    assert "X" not in topo.get_neighbours("A")