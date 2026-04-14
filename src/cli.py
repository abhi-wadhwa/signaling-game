"""
Command-line interface for signaling game analysis.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.beer_quiche import BeerQuicheGame
from src.core.crawford_sobel import CrawfordSobelModel
from src.core.d1_criterion import d1_criterion_filter
from src.core.intuitive_criterion import intuitive_criterion_filter
from src.core.spence import SpenceModel

app = typer.Typer(
    name="signaling-game",
    help="Signaling game equilibrium solver and analyzer.",
)
console = Console()


@app.command()
def spence(
    theta_low: float = typer.Option(1.0, help="Low-type productivity"),
    theta_high: float = typer.Option(2.0, help="High-type productivity"),
    prob_high: float = typer.Option(0.5, help="Prior probability of high type"),
) -> None:
    """Analyze the Spence job market signaling model."""
    model = SpenceModel(
        theta_low=theta_low,
        theta_high=theta_high,
        prob_high=prob_high,
    )

    console.print(Panel("[bold]Spence Job Market Signaling Model[/bold]"))
    console.print(f"  theta_L = {theta_low}, theta_H = {theta_high}")
    console.print(f"  P(high) = {prob_high}, E[theta] = {model.expected_theta:.4f}")
    console.print()

    sep = model.separating_equilibrium()
    console.print(Panel("[bold green]Separating Equilibrium[/bold green]"))
    console.print(sep.description)
    console.print()

    pool = model.pooling_equilibrium()
    console.print(Panel("[bold yellow]Pooling Equilibrium[/bold yellow]"))
    console.print(pool.description)


@app.command()
def crawford_sobel(
    bias: float = typer.Option(0.1, help="Sender bias parameter b"),
) -> None:
    """Analyze the Crawford-Sobel cheap talk model."""
    model = CrawfordSobelModel(bias=bias)
    n_max = model.max_partitions()

    console.print(Panel("[bold]Crawford-Sobel Cheap Talk Model[/bold]"))
    console.print(f"  Bias b = {bias}")
    console.print(f"  Maximum partitions N*(b) = {n_max}")
    console.print()

    equilibria = model.all_partition_equilibria()

    table = Table(title="Partition Equilibria")
    table.add_column("N", style="cyan")
    table.add_column("Boundaries")
    table.add_column("Actions")
    table.add_column("Sender EU", style="green")
    table.add_column("Receiver EU", style="green")

    for eq in equilibria:
        table.add_row(
            str(eq.num_partitions),
            ", ".join(f"{b:.3f}" for b in eq.boundaries),
            ", ".join(f"{a:.3f}" for a in eq.actions),
            f"{eq.sender_eu:.6f}",
            f"{eq.receiver_eu:.6f}",
        )

    console.print(table)


@app.command()
def beer_quiche(
    prob_tough: float = typer.Option(0.9, help="Prior probability of Tough type"),
    apply_ic: bool = typer.Option(False, "--ic", help="Apply Intuitive Criterion"),
    apply_d1: bool = typer.Option(False, "--d1", help="Apply D1 Criterion"),
) -> None:
    """Analyze the Beer-Quiche signaling game."""
    bq = BeerQuicheGame(prob_tough=prob_tough)
    game = bq.to_signaling_game()

    console.print(Panel("[bold]Beer-Quiche Signaling Game[/bold]"))
    console.print(f"  P(Tough) = {prob_tough}")
    console.print()

    pbe_list = bq.enumerate_all_pbe()
    console.print(f"Found [bold]{len(pbe_list)}[/bold] PBE total")
    console.print()

    for i, pbe in enumerate(pbe_list):
        console.print(f"  PBE {i + 1}: {pbe.label} [{pbe.equilibrium_type}]")
        for t in range(2):
            payoff = pbe.sender_payoffs.get(t, 0)
            console.print(f"    Type {game.types[t].name} payoff: {payoff:.2f}")

    if apply_ic:
        console.print()
        surviving = intuitive_criterion_filter(pbe_list, game)
        console.print(
            f"After Intuitive Criterion: [bold]{len(surviving)}[/bold] PBE survive"
        )
        for pbe in surviving:
            console.print(f"  Surviving: {pbe.label}")

    if apply_d1:
        console.print()
        surviving = d1_criterion_filter(pbe_list, game)
        console.print(
            f"After D1 Criterion: [bold]{len(surviving)}[/bold] PBE survive"
        )
        for pbe in surviving:
            console.print(f"  Surviving: {pbe.label}")


@app.command()
def demo() -> None:
    """Run a quick demonstration of all models."""
    console.print(Panel("[bold magenta]Signaling Game Solver Demo[/bold magenta]"))
    console.print()

    # Spence
    console.print("[bold]1. Spence Job Market Signaling[/bold]")
    model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
    sep = model.separating_equilibrium()
    console.print(f"   Separating: e_L=0, e_H={sep.education_levels['high']:.2f}")
    console.print(f"   Wages: w_L={sep.wages['low']:.2f}, w_H={sep.wages['high']:.2f}")
    console.print()

    # Crawford-Sobel
    console.print("[bold]2. Crawford-Sobel Cheap Talk[/bold]")
    cs = CrawfordSobelModel(bias=0.1)
    console.print(f"   Bias b=0.1, max partitions N*={cs.max_partitions()}")
    most = cs.most_informative_equilibrium()
    console.print(f"   Most informative: {most.num_partitions} intervals")
    console.print()

    # Beer-Quiche
    console.print("[bold]3. Beer-Quiche Game[/bold]")
    bq = BeerQuicheGame(prob_tough=0.9)
    game = bq.to_signaling_game()
    pbes = bq.enumerate_pure_pbe()
    console.print(f"   Found {len(pbes)} pure PBE")
    surviving = intuitive_criterion_filter(pbes, game)
    console.print(f"   After Intuitive Criterion: {len(surviving)} survive")
    for pbe in surviving:
        console.print(f"   Surviving: {pbe.label}")


if __name__ == "__main__":
    app()
