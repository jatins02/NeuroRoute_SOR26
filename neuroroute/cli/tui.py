from __future__ import annotations

import asyncio
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

__version__ = "0.1.0"

console = Console()


# --------------------------------------------------------------------------
# TUI Dashboard State
# --------------------------------------------------------------------------

@dataclass
class TUIState:
    """Maintains real-time telemetry state for the TUI dashboard."""

    nodes: List[str] = field(default_factory=lambda: ["nodeA", "nodeB", "nodeC", "nodeD"])
    links: List[Tuple[str, str, float, float]] = field(
        default_factory=lambda: [
            ("nodeA", "nodeB", 40.0, 100.0),
            ("nodeA", "nodeC", 50.0, 120.0),
            ("nodeB", "nodeC", 20.0, 80.0),
            ("nodeB", "nodeD", 30.0, 150.0),
            ("nodeC", "nodeD", 20.0, 100.0),
        ]
    )
    queue_depths: Dict[str, float] = field(
        default_factory=lambda: {"nodeA": 15.0, "nodeB": 42.0, "nodeC": 8.0, "nodeD": 65.0}
    )
    queue_capacities: Dict[str, int] = field(
        default_factory=lambda: {"nodeA": 100, "nodeB": 100, "nodeC": 100, "nodeD": 100}
    )
    packets_generated: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0
    total_latency_ms: float = 0.0
    throughput_pps: float = 0.0
    start_time: float = field(default_factory=time.time)
    event_logs: List[Tuple[str, str]] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.event_logs.append((timestamp, message))
        if len(self.event_logs) > 50:
            self.event_logs.pop(0)

    @property
    def delivery_rate(self) -> float:
        if self.packets_generated == 0:
            return 100.0
        return (self.packets_delivered / self.packets_generated) * 100.0

    @property
    def average_latency(self) -> float:
        if self.packets_delivered == 0:
            return 0.0
        return self.total_latency_ms / self.packets_delivered

    @property
    def runtime(self) -> float:
        return max(0.1, time.time() - self.start_time)


# --------------------------------------------------------------------------
# Render Component Functions
# --------------------------------------------------------------------------

def make_header_panel() -> Panel:
    """Render top header banner."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right", ratio=1)
    grid.add_row(
        "[bold green]NeuroRoute Simulator Dashboard[/bold green]",
        f"[bold cyan]v{__version__}[/bold cyan] | [bold yellow]LIVE[/bold yellow]",
    )
    return Panel(grid, style="bold white on blue")


def make_topology_panel(state: TUIState) -> Panel:
    """Render Network Topology & Link Status panel."""
    table = Table(expand=True, box=None, show_header=True, header_style="bold magenta")
    table.add_column("From", style="cyan")
    table.add_column("To", style="cyan")
    table.add_column("Latency", justify="right", style="yellow")
    table.add_column("Bandwidth", justify="right", style="green")
    table.add_column("Status", justify="center", style="bold green")

    for src, dst, lat, bw in state.links:
        table.add_row(src, dst, f"{lat:.1f} ms", f"{bw:.0f} Mbps", "ONLINE")

    return Panel(table, title="[bold green]Network Topology[/bold green]", border_style="cyan")


def make_metrics_panel(state: TUIState) -> Panel:
    """Render Performance Metrics panel."""
    table = Table(expand=True, box=None, show_header=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right", style="bold white")

    table.add_row("Throughput", f"{state.throughput_pps:.1f} pkts/sec")
    table.add_row("Delivery Rate", f"{state.delivery_rate:.1f}%")
    table.add_row("Packets Generated", str(state.packets_generated))
    table.add_row("Packets Delivered", str(state.packets_delivered))
    table.add_row("Packets Dropped", f"[bold red]{state.packets_dropped}[/bold red]")
    table.add_row("Average Latency", f"{state.average_latency:.2f} ms")
    table.add_row("Elapsed Time", f"{state.runtime:.1f} s")

    return Panel(table, title="[bold green]Performance Metrics[/bold green]", border_style="green")


def make_queue_panel(state: TUIState) -> Panel:
    """Render Queue Buffers fill level panel."""
    table = Table(expand=True, box=None, show_header=False)
    table.add_column("Node", style="bold cyan", width=8)
    table.add_column("Buffer Fill", ratio=1)
    table.add_column("Pct", justify="right", width=6)

    for node in state.nodes:
        depth = state.queue_depths.get(node, 0.0)
        cap = state.queue_capacities.get(node, 100)
        pct = min(100.0, max(0.0, (depth / cap) * 100.0))

        bar_len = 15
        filled_len = int((pct / 100.0) * bar_len)
        empty_len = bar_len - filled_len

        if pct < 60:
            color = "green"
        elif pct < 85:
            color = "yellow"
        else:
            color = "red"

        bar_str = f"[{color}]" + "█" * filled_len + "░" * empty_len + f"[/{color}]"
        table.add_row(node, bar_str, f"[{color}]{pct:.0f}%[/{color}]")

    return Panel(table, title="[bold green]Queue Buffers[/bold green]", border_style="yellow")


def make_event_log_panel(state: TUIState, max_entries: int = 8) -> Panel:
    """Render recent Event Log panel."""
    logs_to_show = state.event_logs[-max_entries:] if state.event_logs else [("", "System initialized.")]
    lines = []
    for ts, msg in logs_to_show:
        if ts:
            lines.append(f"[dim]{ts}[/dim] [white]{msg}[/white]")
        else:
            lines.append(f"[white]{msg}[/white]")

    log_text = Text.from_markup("\n".join(lines))
    return Panel(log_text, title="[bold green]Event Log[/bold green]", border_style="magenta")


# --------------------------------------------------------------------------
# Dashboard Layout Construction
# --------------------------------------------------------------------------

def make_dashboard_layout(state: TUIState, console_height: int = 24) -> Layout:
    """
    Construct top-level multi-panel dashboard layout.
    Dynamically adapts log panel depth based on terminal height.
    """
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
    )

    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )

    layout["left"].split_column(
        Layout(name="topology", ratio=1),
        Layout(name="metrics", ratio=1),
    )

    layout["right"].split_column(
        Layout(name="queues", ratio=1),
        Layout(name="logs", ratio=1),
    )

    layout["header"].update(make_header_panel())
    layout["topology"].update(make_topology_panel(state))
    layout["metrics"].update(make_metrics_panel(state))
    layout["queues"].update(make_queue_panel(state))

    max_logs = max(4, (console_height - 12) // 3)
    layout["logs"].update(make_event_log_panel(state, max_entries=max_logs))

    return layout


# --------------------------------------------------------------------------
# Mock Telemetry Generator & Real-time Render Loop
# --------------------------------------------------------------------------

async def mock_telemetry_loop(
    state: TUIState, refresh_rate: float = 0.25, duration: Optional[int] = None
) -> None:
    """Simulates dynamic network telemetry updates for demonstration."""
    state.add_log("Simulator TUI started.")
    state.add_log("Loaded topology: 4 nodes, 5 bidirectional links.")

    start_time = time.time()
    step_count = 0

    while True:
        if duration and (time.time() - start_time) >= duration:
            break

        step_count += 1
        new_pkts = random.randint(3, 8)
        delivered_pkts = random.randint(2, new_pkts)
        dropped_pkts = new_pkts - delivered_pkts if random.random() < 0.15 else 0

        state.packets_generated += new_pkts
        state.packets_delivered += delivered_pkts
        state.packets_dropped += dropped_pkts
        state.total_latency_ms += delivered_pkts * random.uniform(15.0, 45.0)
        state.throughput_pps = state.packets_delivered / state.runtime

        for node in state.nodes:
            current = state.queue_depths[node]
            delta = random.uniform(-10.0, 12.0)
            state.queue_depths[node] = min(95.0, max(5.0, current + delta))

        if step_count % 3 == 0:
            src = random.choice(state.nodes)
            dst = random.choice([n for n in state.nodes if n != src])
            state.add_log(f"Routed batch {new_pkts} pkts ({src} -> {dst})")

        if dropped_pkts > 0:
            congested_node = random.choice(state.nodes)
            state.add_log(f"[bold red]Buffer overflow at {congested_node}! Dropped {dropped_pkts} pkts.[/bold red]")

        await asyncio.sleep(refresh_rate)


async def run_tui(refresh_rate: float = 0.25, duration: Optional[int] = None) -> None:
    """Run interactive TUI dashboard using Rich Live rendering."""
    state = TUIState()
    telemetry_task = asyncio.create_task(
        mock_telemetry_loop(state, refresh_rate=refresh_rate, duration=duration)
    )

    try:
        with Live(
            make_dashboard_layout(state, console.height),
            console=console,
            refresh_per_second=int(1 / refresh_rate),
            screen=True,
        ) as live:
            while not telemetry_task.done():
                layout = make_dashboard_layout(state, console.height)
                live.update(layout)
                await asyncio.sleep(refresh_rate)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        telemetry_task.cancel()
        console.print("[bold yellow]TUI Dashboard exited cleanly.[/bold yellow]")


# --------------------------------------------------------------------------
# CLI Entrypoint
# --------------------------------------------------------------------------

@click.command(name="tui")
@click.option(
    "-r",
    "--refresh-rate",
    "refresh_rate",
    type=float,
    default=0.25,
    show_default=True,
    help="Dashboard refresh interval in seconds.",
)
@click.option(
    "-d",
    "--duration",
    "duration",
    type=int,
    default=None,
    help="Duration in seconds to run dashboard (default: run until Ctrl+C).",
)
def main(refresh_rate: float, duration: Optional[int]) -> None:
    """Launch the interactive NeuroRoute TUI dashboard."""
    try:
        asyncio.run(run_tui(refresh_rate=refresh_rate, duration=duration))
    except KeyboardInterrupt:
        console.print("[bold yellow]\nDashboard stopped by user.[/bold yellow]")


if __name__ == "__main__":
    main()
