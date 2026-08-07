


from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.traceback import install as install_rich_traceback

__version__ = "0.1.0"

DEFAULT_TOPOLOGY_PATH = "configs/default_topology.yaml"

console = Console()


_VERBOSITY_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def configure_logging(verbosity: int) -> logging.Logger:
   
    level = _VERBOSITY_LEVELS.get(verbosity, logging.DEBUG)

    # Pretty tracebacks for any uncaught exceptions.
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
        force=True,  # Re-configure cleanly if this is called more than once.
    )

    logger = logging.getLogger("neuroroute")
    logger.setLevel(level)
    return logger


def print_banner(topology: str, steps: int) -> None:
    
    body = (
        f"[bold cyan]Topology[/bold cyan]: {topology}\n"
        f"[bold cyan]Steps[/bold cyan]:    {steps}"
    )
    console.print(
        Panel(
            body,
            title="[bold green]NeuroRoute Simulator[/bold green]",
            subtitle=f"v{__version__}",
            expand=False,
        )
    )


@click.command(name="simulate")
@click.option(
    "-t",
    "--topology",
    "topology",
    type=click.Path(),
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
    "-v",
    "--verbose",
    "verbose",
    count=True,
    help="Increase logging verbosity. Use -v for INFO, -vv for DEBUG.",
)
@click.version_option(version=__version__, prog_name="neuroroute-simulate")
def main(topology: str, steps: int, verbose: int) -> None:
    
    logger = configure_logging(verbose)

    print_banner(topology, steps)

    logger.debug("Debug logging enabled.")
    logger.info("Loading topology from '%s'...", topology)
    logger.info("Preparing to run %d simulation steps.", steps)

   
    try:
        logger.warning(
            "Simulation engine not yet implemented (see later issues)."
        )
    except Exception:  # pragma: no cover - placeholder safety net
        logger.exception("Simulation failed unexpectedly.")
        sys.exit(1)

    logger.info("Done.")


# hello

if __name__ == "__main__":
    main()