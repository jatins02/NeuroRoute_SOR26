import asyncio
import pytest
from rich.layout import Layout
from rich.panel import Panel

from neuroroute.cli.tui import (
    TUIState,
    cycle_strategy,
    make_dashboard_layout,
    make_event_log_panel,
    make_header_panel,
    make_latency_chart_panel,
    make_metrics_panel,
    make_queue_panel,
    make_topology_panel,
    mock_telemetry_loop,
)
from neuroroute.network.topology import TopologyManager
from neuroroute.router.plane import SimRouterNode
from neuroroute.network.algorithms import Dijkstras


def test_tui_state_initialization():
    state = TUIState()
    assert len(state.nodes) == 4
    assert len(state.links) == 5
    assert state.packets_generated == 0
    assert state.delivery_rate == 100.0
    assert state.average_latency == 0.0
    assert state.runtime > 0.0


def test_tui_state_add_log():
    state = TUIState()
    state.add_log("Test message 1")
    assert len(state.event_logs) == 1
    assert "Test message 1" in state.event_logs[0][1]

    for i in range(60):
        state.add_log(f"Msg {i}")

    assert len(state.event_logs) == 50
    assert "Msg 59" in state.event_logs[-1][1]


def test_panel_renderers():
    state = TUIState()
    state.packets_generated = 10
    state.packets_delivered = 8
    state.packets_dropped = 2
    state.total_latency_ms = 160.0
    state.latency_history = [20.0, 25.0, 30.0, 15.0]

    header = make_header_panel(state)
    assert isinstance(header, Panel)

    topo = make_topology_panel(state)
    assert isinstance(topo, Panel)

    metrics = make_metrics_panel(state)
    assert isinstance(metrics, Panel)

    queues = make_queue_panel(state)
    assert isinstance(queues, Panel)

    chart = make_latency_chart_panel(state)
    assert isinstance(chart, Panel)

    logs = make_event_log_panel(state)
    assert isinstance(logs, Panel)


def test_make_dashboard_layout():
    state = TUIState()
    layout = make_dashboard_layout(state, console_height=30)
    assert isinstance(layout, Layout)
    assert layout["header"].name == "header"
    assert layout["body"].name == "body"
    assert layout["chart"].name == "chart"


def test_cycle_strategy():
    state = TUIState(current_strategy="static")
    new_strat = cycle_strategy(state)
    assert new_strat == "round-robin"
    assert state.current_strategy == "round-robin"


def test_update_from_simulation():
    topo = TopologyManager()
    topo.graph = {"nodeA": {"nodeB": {"latency": 10.0, "bandwidth": 100.0}}}
    strategy = Dijkstras(topo)
    node = SimRouterNode("nodeA", topo, strategy)

    stats = {
        "packets_generated": 15,
        "packets_delivered": 12,
        "packets_dropped": 3,
        "total_latency": 0.36,
    }

    state = TUIState(
        router_nodes={"nodeA": node},
        topology_manager=topo,
        stats_ref=stats,
    )

    state.update_from_simulation()

    assert state.packets_generated == 15
    assert state.packets_delivered == 12
    assert state.packets_dropped == 3
    assert len(state.latency_history) == 1


def test_mock_telemetry_loop_short_run():
    state = TUIState()
    asyncio.run(mock_telemetry_loop(state, refresh_rate=0.01, duration=0.1))

    assert state.packets_generated > 0
    assert state.packets_delivered > 0
    assert len(state.event_logs) > 0
