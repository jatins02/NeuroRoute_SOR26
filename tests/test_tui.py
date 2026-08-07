import asyncio
import pytest
from rich.layout import Layout
from rich.panel import Panel

from neuroroute.cli.tui import (
    TUIState,
    make_dashboard_layout,
    make_event_log_panel,
    make_header_panel,
    make_metrics_panel,
    make_queue_panel,
    make_topology_panel,
    mock_telemetry_loop,
)


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

    # Test log list capping
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

    header = make_header_panel()
    assert isinstance(header, Panel)

    topo = make_topology_panel(state)
    assert isinstance(topo, Panel)

    metrics = make_metrics_panel(state)
    assert isinstance(metrics, Panel)

    queues = make_queue_panel(state)
    assert isinstance(queues, Panel)

    logs = make_event_log_panel(state)
    assert isinstance(logs, Panel)


def test_make_dashboard_layout():
    state = TUIState()
    layout = make_dashboard_layout(state, console_height=30)
    assert isinstance(layout, Layout)
    assert layout["header"].name == "header"
    assert layout["body"].name == "body"


def test_mock_telemetry_loop_short_run():
    state = TUIState()
    # Run mock telemetry loop for 0.1 seconds
    asyncio.run(mock_telemetry_loop(state, refresh_rate=0.01, duration=0.1))

    assert state.packets_generated > 0
    assert state.packets_delivered > 0
    assert len(state.event_logs) > 0

