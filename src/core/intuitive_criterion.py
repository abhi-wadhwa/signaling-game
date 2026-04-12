"""
Intuitive Criterion (Cho & Kreps, 1987).

The Intuitive Criterion refines the set of PBE by restricting off-path beliefs.

Idea: If a signal m is off the equilibrium path, consider which types could
possibly benefit from deviating to m. If sending m is "equilibrium dominated"
for some type theta (i.e., even the best possible response the receiver could
give after seeing m leaves theta worse off than their equilibrium payoff),
then the receiver should place zero belief on that type after seeing m.

Formally, a PBE fails the Intuitive Criterion if there exists an off-path
signal m and a type theta such that:
  1. Signal m is equilibrium dominated for all types EXCEPT theta
  2. Type theta would strictly benefit from sending m if the receiver
     held belief mu(theta|m) = 1

This eliminates "unreasonable" PBE supported by implausible off-path beliefs.
"""

from __future__ import annotations

from src.core.signaling import SignalingGame, PBE


def _equilibrium_payoff(pbe: PBE, game: SignalingGame, type_idx: int) -> float:
    """Compute type's equilibrium payoff."""
    eu = 0.0
    for m in range(game.num_signals):
        sigma_m = pbe.sender_strategy.get((type_idx, m), 0.0)
        if sigma_m < 1e-12:
            continue
        for a in range(game.num_actions):
            alpha_a = pbe.receiver_strategy.get((m, a), 0.0)
            eu += sigma_m * alpha_a * game.u_s(type_idx, m, a)
    return eu


def _max_deviation_payoff(
    game: SignalingGame, type_idx: int, signal_idx: int
) -> float:
    """
    Maximum payoff type could get by deviating to signal, over all
    possible receiver responses.

    max_a u_S(type_idx, signal_idx, a)
    """
    return max(
        game.u_s(type_idx, signal_idx, a)
        for a in range(game.num_actions)
    )


def _best_response_payoff_under_belief(
    game: SignalingGame,
    type_idx: int,
    signal_idx: int,
    belief: dict[int, float],
) -> float:
    """
    Payoff to type_idx from signal_idx when receiver best-responds to belief.
    """
    # Find receiver's best response under belief
    best_a = -1
    best_eu_r = float("-inf")
    for a in range(game.num_actions):
        eu_r = game.expected_receiver_payoff(signal_idx, a, belief)
        if eu_r > best_eu_r:
            best_eu_r = eu_r
            best_a = a

    return game.u_s(type_idx, signal_idx, best_a)


def check_intuitive_criterion(pbe: PBE, game: SignalingGame) -> bool:
    """
    Check if a PBE satisfies the Intuitive Criterion.

    Returns True if the PBE passes (survives the refinement).
    Returns False if it fails (should be eliminated).
    """
    for m in range(game.num_signals):
        if pbe.is_on_path(m):
            continue  # Only check off-path signals

        # Find types for which m is equilibrium dominated
        eq_dominated_types: set[int] = set()
        non_dominated_types: set[int] = set()

        for t in range(game.num_types):
            eq_payoff = _equilibrium_payoff(pbe, game, t)
            max_dev = _max_deviation_payoff(game, t, m)

            if max_dev < eq_payoff - 1e-9:
                eq_dominated_types.add(t)
            else:
                non_dominated_types.add(t)

        if not non_dominated_types:
            continue  # m is dominated for all types, no issue

        # If only one type is non-dominated, check if that type benefits
        # from deviating when receiver believes it's that type
        if len(non_dominated_types) == 1:
            theta = next(iter(non_dominated_types))
            # Under belief mu(theta|m) = 1, receiver best-responds
            belief = {t: (1.0 if t == theta else 0.0) for t in range(game.num_types)}
            dev_payoff = _best_response_payoff_under_belief(game, theta, m, belief)
            eq_payoff = _equilibrium_payoff(pbe, game, theta)

            if dev_payoff > eq_payoff + 1e-9:
                return False  # Fails Intuitive Criterion

        # More general: if belief concentrated on non-dominated types
        # would make some non-dominated type want to deviate
        if len(non_dominated_types) >= 2:
            # Check each non-dominated type with belief concentrated on it
            for theta in non_dominated_types:
                belief = {t: (1.0 if t == theta else 0.0) for t in range(game.num_types)}
                dev_payoff = _best_response_payoff_under_belief(game, theta, m, belief)
                eq_payoff = _equilibrium_payoff(pbe, game, theta)

                # Check if ALL non-dominated types besides theta are also dominated
                # under the restriction
                others_dominated = True
                for t2 in non_dominated_types:
                    if t2 == theta:
                        continue
                    # Would t2 benefit from m under ANY belief restricted to non-dominated?
                    belief_t2 = {t: (1.0 if t == t2 else 0.0) for t in range(game.num_types)}
                    dev_t2 = _best_response_payoff_under_belief(game, t2, m, belief_t2)
                    eq_t2 = _equilibrium_payoff(pbe, game, t2)
                    if dev_t2 > eq_t2 + 1e-9:
                        others_dominated = False

                if others_dominated and dev_payoff > eq_payoff + 1e-9:
                    return False

    return True


def intuitive_criterion_filter(
    pbe_list: list[PBE], game: SignalingGame
) -> list[PBE]:
    """
    Filter a list of PBE by the Intuitive Criterion.

    Returns only those PBE that pass (survive) the refinement.
    """
    return [pbe for pbe in pbe_list if check_intuitive_criterion(pbe, game)]
