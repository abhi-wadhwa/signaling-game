"""
D1 Criterion (Banks & Sobel, 1987; Cho & Kreps, 1987).

The D1 Criterion is a stronger refinement than the Intuitive Criterion.

Intuitive Criterion: eliminates PBE where a type is dominated for deviation
D1: eliminates PBE where, for an off-path signal m, the set of receiver
    responses that make type theta willing to deviate is a STRICT SUBSET
    of those for another type theta'.

Formally: For off-path signal m, define
    D(theta, m) = {a : u_S(theta, m, a) > u_S*(theta)}  (strict improvement)
    D0(theta, m) = {a : u_S(theta, m, a) >= u_S*(theta)}  (weak improvement)

D1 says: if D0(theta, m) is a SUBSET of D(theta', m) for some theta' != theta,
then mu(theta | m) = 0.

After restricting beliefs, check if remaining PBE is still an equilibrium.
"""

from __future__ import annotations

from src.core.signaling import SignalingGame, PBE
from src.core.intuitive_criterion import _equilibrium_payoff


def _compute_d_sets(
    game: SignalingGame,
    pbe: PBE,
    type_idx: int,
    signal_idx: int,
) -> tuple[set[int], set[int]]:
    """
    Compute D(theta, m) and D0(theta, m).

    D(theta, m) = {a : u_S(theta, m, a) > u_S*(theta)}
    D0(theta, m) = {a : u_S(theta, m, a) >= u_S*(theta)}
    """
    eq_payoff = _equilibrium_payoff(pbe, game, type_idx)

    d_strict: set[int] = set()
    d_weak: set[int] = set()

    for a in range(game.num_actions):
        dev_payoff = game.u_s(type_idx, signal_idx, a)
        if dev_payoff > eq_payoff + 1e-9:
            d_strict.add(a)
            d_weak.add(a)
        elif dev_payoff >= eq_payoff - 1e-9:
            d_weak.add(a)

    return d_strict, d_weak


def check_d1_criterion(pbe: PBE, game: SignalingGame) -> bool:
    """
    Check if a PBE satisfies the D1 criterion.

    Returns True if the PBE passes (survives).
    Returns False if it fails (should be eliminated).
    """
    for m in range(game.num_signals):
        if pbe.is_on_path(m):
            continue

        # Compute D and D0 for all types
        d_strict: dict[int, set[int]] = {}
        d_weak: dict[int, set[int]] = {}

        for t in range(game.num_types):
            ds, dw = _compute_d_sets(game, pbe, t, m)
            d_strict[t] = ds
            d_weak[t] = dw

        # Check D1 condition: for each type theta, check if D0(theta, m) subset D(theta', m)
        eliminated_types: set[int] = set()
        for t in range(game.num_types):
            for t_prime in range(game.num_types):
                if t_prime == t:
                    continue
                # D1: if D0(t, m) is a subset of D(t', m), eliminate t
                if d_weak[t] <= d_strict[t_prime]:  # set subset
                    eliminated_types.add(t)
                    break

        remaining_types = set(range(game.num_types)) - eliminated_types

        if not remaining_types:
            continue  # All types eliminated, signal not rationalizable

        # Under D1 beliefs (restricted to remaining types), check if
        # any remaining type wants to deviate
        if len(remaining_types) == 1:
            theta = next(iter(remaining_types))
            # Under belief mu(theta|m) = 1
            # Find receiver BR
            belief = {t: (1.0 if t == theta else 0.0) for t in range(game.num_types)}
            best_a = _receiver_br(game, m, belief)
            dev_payoff = game.u_s(theta, m, best_a)
            eq_payoff = _equilibrium_payoff(pbe, game, theta)

            if dev_payoff > eq_payoff + 1e-9:
                return False  # Fails D1

        elif len(remaining_types) > 1:
            # Check with uniform beliefs over remaining types
            total = len(remaining_types)
            belief = {
                t: (1.0 / total if t in remaining_types else 0.0)
                for t in range(game.num_types)
            }
            best_a = _receiver_br(game, m, belief)

            for theta in remaining_types:
                dev_payoff = game.u_s(theta, m, best_a)
                eq_payoff = _equilibrium_payoff(pbe, game, theta)
                if dev_payoff > eq_payoff + 1e-9:
                    return False

    return True


def _receiver_br(
    game: SignalingGame, signal_idx: int, belief: dict[int, float]
) -> int:
    """Find receiver's best response action given beliefs."""
    best_a = 0
    best_eu = float("-inf")
    for a in range(game.num_actions):
        eu = game.expected_receiver_payoff(signal_idx, a, belief)
        if eu > best_eu:
            best_eu = eu
            best_a = a
    return best_a


def d1_criterion_filter(
    pbe_list: list[PBE], game: SignalingGame
) -> list[PBE]:
    """
    Filter a list of PBE by the D1 criterion.

    Returns only those PBE that pass (survive) the refinement.
    D1 is stronger than Intuitive Criterion, so D1-surviving is a subset.
    """
    return [pbe for pbe in pbe_list if check_d1_criterion(pbe, game)]
