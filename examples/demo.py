"""
Demonstration of the signaling game solver.

Run: python -m examples.demo
"""

from src.core.spence import SpenceModel
from src.core.crawford_sobel import CrawfordSobelModel
from src.core.beer_quiche import BeerQuicheGame, TOUGH, WEAK, BEER, QUICHE
from src.core.pbe_solver import PBESolver
from src.core.intuitive_criterion import intuitive_criterion_filter
from src.core.d1_criterion import d1_criterion_filter


def demo_spence() -> None:
    """Demonstrate the Spence job market signaling model."""
    print("=" * 70)
    print("SPENCE JOB MARKET SIGNALING MODEL")
    print("=" * 70)

    model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)

    print(f"\nParameters: theta_L={model.theta_low}, theta_H={model.theta_high}, "
          f"P(H)={model.prob_high}")
    print(f"E[theta] = {model.expected_theta:.4f}")
    print(f"Cost function: c(e, theta) = e / theta")

    # Separating equilibrium
    sep = model.separating_equilibrium()
    print(f"\n{sep.description}")

    # Pooling equilibrium
    pool = model.pooling_equilibrium()
    print(f"\n{pool.description}")

    print()


def demo_crawford_sobel() -> None:
    """Demonstrate the Crawford-Sobel cheap talk model."""
    print("=" * 70)
    print("CRAWFORD-SOBEL CHEAP TALK MODEL")
    print("=" * 70)

    for bias in [0.05, 0.1, 0.2]:
        model = CrawfordSobelModel(bias=bias)
        n_star = model.max_partitions()

        print(f"\nBias b={bias}, N*(b)={n_star}")

        for n in range(1, n_star + 1):
            eq = model.partition_equilibrium(n)
            if eq is not None:
                boundaries = [f"{b:.3f}" for b in eq.boundaries]
                actions = [f"{a:.3f}" for a in eq.actions]
                print(f"  N={n}: boundaries={boundaries}, "
                      f"receiver_EU={eq.receiver_eu:.6f}")

    print()


def demo_beer_quiche() -> None:
    """Demonstrate the Beer-Quiche game with refinements."""
    print("=" * 70)
    print("BEER-QUICHE SIGNALING GAME")
    print("=" * 70)

    bq = BeerQuicheGame(prob_tough=0.9)
    game = bq.to_signaling_game()

    print(f"\nP(Tough) = {bq.prob_tough}")

    # Enumerate all PBE
    all_pbe = bq.enumerate_all_pbe()
    print(f"\nFound {len(all_pbe)} total PBE:")

    for i, pbe in enumerate(all_pbe):
        print(f"\n  PBE {i+1}: {pbe.label} [{pbe.equilibrium_type}]")
        for t in [TOUGH, WEAK]:
            print(f"    Type {game.types[t].name}: payoff = {pbe.sender_payoffs.get(t, 0):.2f}")

    # Apply refinements
    ic_survivors = intuitive_criterion_filter(all_pbe, game)
    print(f"\nAfter Intuitive Criterion: {len(ic_survivors)} PBE survive")
    for pbe in ic_survivors:
        print(f"  Survives: {pbe.label}")

    d1_survivors = d1_criterion_filter(all_pbe, game)
    print(f"\nAfter D1 Criterion: {len(d1_survivors)} PBE survive")
    for pbe in d1_survivors:
        print(f"  Survives: {pbe.label}")

    print()


def demo_general_solver() -> None:
    """Demonstrate the general PBE solver."""
    print("=" * 70)
    print("GENERAL PBE SOLVER")
    print("=" * 70)

    # Use Beer-Quiche as the test game
    bq = BeerQuicheGame(prob_tough=0.9)
    game = bq.to_signaling_game()
    solver = PBESolver(game)

    print("\nSolving Beer-Quiche via general PBE solver...")
    pbes = solver.find_pure_pbe()
    print(f"Found {len(pbes)} pure-strategy PBE")

    for i, pbe in enumerate(pbes):
        print(f"  PBE {i+1}: {pbe.label} [{pbe.equilibrium_type}]")

    print()


def demo_belief_updating() -> None:
    """Demonstrate Bayesian belief updating."""
    print("=" * 70)
    print("BAYESIAN BELIEF UPDATING")
    print("=" * 70)

    bq = BeerQuicheGame(prob_tough=0.9)
    game = bq.to_signaling_game()

    print(f"\nPrior: P(Tough)={game.prior[TOUGH]:.2f}, P(Weak)={game.prior[WEAK]:.2f}")

    # Separating strategy: Tough->Beer, Weak->Quiche
    sender_strat = {
        (TOUGH, BEER): 1.0, (TOUGH, QUICHE): 0.0,
        (WEAK, BEER): 0.0, (WEAK, QUICHE): 1.0,
    }

    print("\nSeparating strategy (Tough->Beer, Weak->Quiche):")
    for m, name in [(BEER, "Beer"), (QUICHE, "Quiche")]:
        posterior = game.bayes_update(m, sender_strat)
        p_str = ", ".join(f"{game.types[t].name}={p:.4f}" for t, p in posterior.items())
        print(f"  After observing {name}: {p_str}")

    # Pooling strategy: both -> Beer
    sender_strat_pool = {
        (TOUGH, BEER): 1.0, (TOUGH, QUICHE): 0.0,
        (WEAK, BEER): 1.0, (WEAK, QUICHE): 0.0,
    }

    print("\nPooling strategy (both -> Beer):")
    for m, name in [(BEER, "Beer"), (QUICHE, "Quiche")]:
        posterior = game.bayes_update(m, sender_strat_pool)
        if posterior:
            p_str = ", ".join(f"{game.types[t].name}={p:.4f}" for t, p in posterior.items())
            print(f"  After observing {name}: {p_str}")
        else:
            print(f"  After observing {name}: OFF-PATH (beliefs unrestricted)")

    print()


if __name__ == "__main__":
    demo_spence()
    demo_crawford_sobel()
    demo_beer_quiche()
    demo_general_solver()
    demo_belief_updating()
