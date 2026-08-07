from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

from neuroroute.ai.agent import QLearningAgent
from neuroroute.ai.env import NetworkRoutingEnv
from neuroroute.network.algorithms import Dijkstras, Random, RoundRobin
from neuroroute.network.topology import TopologyManager
from neuroroute.router.plane import Packet, SimRouterNode

__version__ = "0.1.0"
DEFAULT_TOPOLOGY_PATH = "configs/square-topology.json"

console = Console()

_VERBOSITY_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def configure_logging(verbosity: int) -> logging.Logger:
    level = _VERBOSITY_LEVELS.get(verbosity, logging.DEBUG)

    install_rich_traceback(console=console, show_locals=verbosity >= 2)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_time=True,
                show_path=verbosity >= 2,
                markup=True,
            )
        ],
        force=True,
    )

    logger = logging.getLogger("neuroroute")
    logger.setLevel(level)
    return logger


def print_banner(topology: str, steps: int, strategy: str) -> None:
    body = (
        f"[bold cyan]Topology[/bold cyan]: {topology}\n"
        f"[bold cyan]Steps[/bold cyan]:    {steps}\n"
        f"[bold cyan]Strategy[/bold cyan]: {strategy}"
    )
    console.print(
        Panel(
            body,
            title="[bold green]NeuroRoute Simulator[/bold green]",
            subtitle=f"v{__version__}",
            expand=False,
        )
    )


class QLearningStrategy:
    """Wrapper strategy around QLearningAgent and NetworkRoutingEnv."""

    def __init__(self, topo: TopologyManager) -> None:
        self.topo = topo
        self.nodes = topo.get_all_nodes()
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.agent = QLearningAgent(
            num_states=len(self.nodes),
            num_actions=len(self.nodes),
            epsilon=0.1,
        )
        self.env = NetworkRoutingEnv(
            num_nodes=len(self.nodes),
            topology_graph=topo,
        )

    def get_next_hop(self, current: str, destination: str) -> Optional[str]:
        if current == destination:
            return current

        if current not in self.node_to_idx or destination not in self.node_to_idx:
            return None

        curr_idx = self.node_to_idx[current]
        action_mask = self.env.get_action_mask(curr_idx)

        action_idx = self.agent.choose_action(curr_idx, valid_actions=action_mask)

        if 0 <= action_idx < len(self.nodes) and action_mask[action_idx]:
            return self.nodes[action_idx]

        neighbours = self.topo.get_neighbours(current)
        return random.choice(neighbours) if neighbours else None


def get_strategy(strategy_name: str, topo: TopologyManager) -> Any:
    name = strategy_name.lower().replace("-", "").replace("_", "")
    if name in ("static", "dijkstra", "dijkstras"):
        return Dijkstras(topo)
    elif name in ("roundrobin", "rr"):
        return RoundRobin(topo)
    elif name == "random":
        return Random(topo)
    elif name in ("qlearning", "qlearningagent", "ql"):
        return QLearningStrategy(topo)
    else:
        raise ValueError(f"Unknown routing strategy: {strategy_name}")


async def _generate_packets(
    nodes: List[str],
    router_nodes: Dict[str, SimRouterNode],
    total_steps: int,
    stats: Dict[str, Any],
    logger: logging.Logger,
    stop_event: asyncio.Event,
) -> None:
    if len(nodes) < 2:
        logger.warning("Network has fewer than 2 nodes. Packet generation skipped.")
        return

    for step in range(total_steps):
        if stop_event.is_set():
            break

        src, dst = random.sample(nodes, 2)
        packet = Packet(
            packet_id="",
            source=src,
            destination=dst,
            payload=f"sim-packet-{step}".encode("utf-8"),
        )

        logger.debug(
            f"Step {step+1}/{total_steps}: Generating packet {packet.packet_id[:8]} ({src} -> {dst})"
        )

        enqueued = await router_nodes[src].enqueue(packet)
        if enqueued:
            stats["packets_generated"] += 1
        else:
            stats["packets_dropped"] += 1
            logger.debug(f"Buffer full at source node {src}. Packet dropped.")

        await asyncio.sleep(0.01)


async def run_simulation(
    topology_path: str,
    steps: int,
    strategy_name: str,
    logger: logging.Logger,
    use_tui: bool = False,
) -> Dict[str, Any]:
    logger.info("Loading topology from '%s'...", topology_path)
    topo = TopologyManager()
    topo.load_topology(topology_path)

    nodes = topo.get_all_nodes()
    if not nodes:
        raise ValueError(f"No nodes found in topology file {topology_path}")

    logger.info("Initializing %d router nodes...", len(nodes))
    strategy = get_strategy(strategy_name, topo)

    router_nodes: Dict[str, SimRouterNode] = {
        node_id: SimRouterNode(node_id, topo, strategy) for node_id in nodes
    }

    for node in router_nodes.values():
        node.set_peers(router_nodes)

    stats: Dict[str, Any] = {
        "packets_generated": 0,
        "packets_delivered": 0,
        "packets_dropped": 0,
        "total_latency": 0.0,
        "start_time": time.time(),
        "end_time": 0.0,
    }

    stop_event = asyncio.Event()

    node_tasks = [
        asyncio.create_task(node.run(stop_event, stats))
        for node in router_nodes.values()
    ]

    tui_task = None
    if use_tui:
        from neuroroute.cli.tui import TUIState, run_live_tui
        links = []
        for src, neighbors in topo.graph.items():
            for dst, metrics in neighbors.items():
                links.append(
                    (src, dst, float(metrics.get("latency", 0.0)), float(metrics.get("bandwidth", 0.0)))
                )

        tui_state = TUIState(
            nodes=nodes,
            links=links,
            current_strategy=strategy_name,
            router_nodes=router_nodes,
            topology_manager=topo,
            stats_ref=stats,
        )
        tui_task = asyncio.create_task(run_live_tui(tui_state, stop_event, refresh_rate=0.1))

    logger.info(
        "Starting packet generation loop (%d steps, strategy: %s)...",
        steps,
        strategy_name,
    )
    await _generate_packets(nodes, router_nodes, steps, stats, logger, stop_event)

    logger.info("Packet generation complete. Draining remaining network queues...")
    await asyncio.sleep(0.5)

    stop_event.set()
    await asyncio.gather(*node_tasks, return_exceptions=True)
    if tui_task:
        await asyncio.gather(tui_task, return_exceptions=True)

    stats["end_time"] = time.time()
    return stats


def display_summary(stats: Dict[str, Any], strategy: str, topology: str) -> None:
    runtime = max(0.001, stats["end_time"] - stats["start_time"])
    delivered = stats["packets_delivered"]
    avg_latency = (
        (stats["total_latency"] / delivered) * 1000.0 if delivered > 0 else 0.0
    )

    table = Table(
        title="Simulation Statistics Summary",
        title_style="bold green",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold white", justify="right")

    table.add_row("Strategy", strategy)
    table.add_row("Topology", topology)
    table.add_row("Packets Generated", str(stats["packets_generated"]))
    table.add_row("Packets Delivered", str(stats["packets_delivered"]))
    table.add_row("Packets Dropped", str(stats["packets_dropped"]))
    table.add_row("Average Latency", f"{avg_latency:.2f} ms")
    table.add_row("Total Runtime", f"{runtime:.2f} s")

    console.print()
    console.print(table)


@click.command(name="simulate")
@click.option(
    "-t",
    "--topology",
    "topology",
    type=click.Path(exists=True),
    default=DEFAULT_TOPOLOGY_PATH,
    show_default=True,
    help="Path to the network topology JSON/YAML file.",
)
@click.option(
    "-s",
    "--steps",
    "steps",
    type=int,
    default=100,
    show_default=True,
    help="Total number of simulation steps to run.",
)
@click.option(
    "-r",
    "--strategy",
    "strategy",
    type=click.Choice(["static", "round-robin", "random", "qlearning"], case_sensitive=False),
    default="static",
    show_default=True,
    help="Routing strategy/algorithm to execute.",
)
@click.option(
    "--tui",
    "use_tui",
    is_flag=True,
    default=False,
    help="Run live TUI dashboard during simulation.",
)
@click.option(
    "-v",
    "--verbose",
    "verbose",
    count=True,
    help="Increase logging verbosity. Use -v for INFO, -vv for DEBUG.",
)
@click.version_option(version=__version__, prog_name="neuroroute-simulate")
def main(topology: str, steps: int, strategy: str, use_tui: bool, verbose: int) -> None:
    logger = configure_logging(verbose)
    if not use_tui:
        print_banner(topology, steps, strategy)

    stats: Optional[Dict[str, Any]] = None
    start_time = time.time()

    try:
        stats = asyncio.run(run_simulation(topology, steps, strategy, logger, use_tui=use_tui))
    except KeyboardInterrupt:
        logger.warning("\n[bold yellow]Simulation interrupted by user (Ctrl+C). Cleaning up...[/bold yellow]")
        if stats is None:
            stats = {
                "packets_generated": 0,
                "packets_delivered": 0,
                "packets_dropped": 0,
                "total_latency": 0.0,
                "start_time": start_time,
                "end_time": time.time(),
            }
        else:
            stats["end_time"] = time.time()
    except Exception as e:
        logger.exception("Simulation failed unexpectedly: %s", e)
        sys.exit(1)

    display_summary(stats, strategy, topology)
    logger.info("Done.")


if __name__ == "__main__":
    main()