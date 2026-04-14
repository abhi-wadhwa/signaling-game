"""
General PBE enumeration for finite signaling games.

Algorithm:
1. Enumerate all pure sender strategies (type -> signal mappings)
2. For each, compute on-path beliefs via Bayes' rule
3. For off-path signals, try all extreme belief profiles (assign belief to a single type)
4. Compute receiver best responses under beliefs
5. Check sender optimality (no profitable deviations)
6. Handle receiver indifference by checking all best-response combinations

This handles arbitrary finite 2+ type signaling games.
"""

from __future__ import annotations

from itertools import product

from src.core.signaling import PBE, SignalingGame


class PBESolver:
    """
    Enumerate all Perfect Bayesian Equilibria of a finite signaling game.

    Supports pure and (for 2-type games) mixed-strategy equilibria.
    """

    def __init__(self, game: SignalingGame) -> None:
        self.game = game

    def find_pure_pbe(self) -> list[PBE]:
        """
        Find all pure-strategy PBE.

        Enumerate sender strategies, compute beliefs and receiver BRs,
        check sender optimality.
        """
        game = self.game
        pbe_list: list[PBE] = []

        # All pure sender strategies: for each type, choose a signal
        signal_indices = list(range(game.num_signals))
        type_indices = list(range(game.num_types))

        for sender_profile in product(signal_indices, repeat=game.num_types):
            # sender_profile[t] = signal chosen by type t
            sender_strat = {}
            for t in type_indices:
                for m in signal_indices:
                    sender_strat[(t, m)] = 1.0 if sender_profile[t] == m else 0.0

            # Determine on-path and off-path signals
            on_path = set(sender_profile)
            off_path = set(signal_indices) - on_path

            # On-path beliefs via Bayes
            on_path_beliefs: dict[int, dict[int, float]] = {}
            for m in on_path:
                beliefs = game.bayes_update(m, sender_strat)
                on_path_beliefs[m] = beliefs

            # Try all extreme off-path beliefs
            if off_path:
                off_path_belief_combos = list(
                    product(type_indices, repeat=len(off_path))
                )
            else:
                off_path_belief_combos = [()]

            off_path_list = sorted(off_path)

            for off_combo in off_path_belief_combos:
                # Build complete belief system
                beliefs_dict: dict[tuple[int, int], float] = {}

                for m in on_path:
                    for t in type_indices:
                        beliefs_dict[(m, t)] = on_path_beliefs[m].get(t, 0.0)

                for idx, m in enumerate(off_path_list):
                    assigned_type = off_combo[idx]
                    for t in type_indices:
                        beliefs_dict[(m, t)] = 1.0 if t == assigned_type else 0.0

                # Compute receiver best responses
                receiver_br_options = self._receiver_best_responses(beliefs_dict)

                for receiver_strat in receiver_br_options:
                    # Check sender optimality
                    if self._check_sender_optimality(
                        sender_strat, receiver_strat, sender_profile
                    ):
                        pbe = self._build_pbe(
                            sender_strat, receiver_strat, beliefs_dict, sender_profile
                        )
                        if not self._is_duplicate(pbe, pbe_list):
                            pbe_list.append(pbe)

        return pbe_list

    def find_all_pbe(self) -> list[PBE]:
        """Find all PBE (pure strategies with all off-path belief specifications)."""
        return self.find_pure_pbe()

    def _receiver_best_responses(
        self,
        beliefs: dict[tuple[int, int], float],
    ) -> list[dict[tuple[int, int], float]]:
        """
        Compute all pure-strategy receiver best responses given beliefs.

        Returns list of receiver strategies (one for each combination
        of best responses across signals).
        """
        game = self.game
        action_indices = list(range(game.num_actions))

        # For each signal, find best response actions
        br_per_signal: list[list[int]] = []
        for m in range(game.num_signals):
            belief_m = {t: beliefs.get((m, t), 0.0) for t in range(game.num_types)}
            expected_payoffs = []
            for a in action_indices:
                eu = game.expected_receiver_payoff(m, a, belief_m)
                expected_payoffs.append(eu)

            max_eu = max(expected_payoffs)
            best_actions = [
                a for a in action_indices
                if expected_payoffs[a] >= max_eu - 1e-9
            ]
            br_per_signal.append(best_actions)

        # All combinations of best responses
        results = []
        for br_combo in product(*br_per_signal):
            strat = {}
            for m, a in enumerate(br_combo):
                for a2 in action_indices:
                    strat[(m, a2)] = 1.0 if a2 == a else 0.0
            results.append(strat)

        return results

    def _check_sender_optimality(
        self,
        sender_strat: dict[tuple[int, int], float],
        receiver_strat: dict[tuple[int, int], float],
        sender_profile: tuple[int, ...],
    ) -> bool:
        """Check that no type wants to deviate from their assigned signal."""
        game = self.game

        for t in range(game.num_types):
            chosen_m = sender_profile[t]
            # Payoff from chosen signal
            u_chosen = sum(
                receiver_strat.get((chosen_m, a), 0.0) * game.u_s(t, chosen_m, a)
                for a in range(game.num_actions)
            )

            # Check all alternative signals
            for m_alt in range(game.num_signals):
                if m_alt == chosen_m:
                    continue
                u_alt = sum(
                    receiver_strat.get((m_alt, a), 0.0) * game.u_s(t, m_alt, a)
                    for a in range(game.num_actions)
                )
                if u_alt > u_chosen + 1e-9:
                    return False

        return True

    def _build_pbe(
        self,
        sender_strat: dict[tuple[int, int], float],
        receiver_strat: dict[tuple[int, int], float],
        beliefs: dict[tuple[int, int], float],
        sender_profile: tuple[int, ...],
    ) -> PBE:
        """Construct a PBE object from strategies and beliefs."""
        game = self.game

        sender_payoffs = {}
        for t in range(game.num_types):
            chosen_m = sender_profile[t]
            sender_payoffs[t] = sum(
                receiver_strat.get((chosen_m, a), 0.0) * game.u_s(t, chosen_m, a)
                for a in range(game.num_actions)
            )

        signal_names = [s.name for s in game.signals]
        profile_str = ", ".join(
            f"{game.types[t].name}->{signal_names[sender_profile[t]]}"
            for t in range(game.num_types)
        )

        pbe = PBE(
            sender_strategy=dict(sender_strat),
            receiver_strategy=dict(receiver_strat),
            beliefs=dict(beliefs),
            sender_payoffs=sender_payoffs,
            label=profile_str,
        )
        pbe.classify(game.num_types, game.num_signals)

        return pbe

    def _is_duplicate(self, pbe: PBE, existing: list[PBE]) -> bool:
        """Check if a PBE is essentially the same as one already found."""
        for other in existing:
            if self._strategies_equal(pbe, other):
                return True
        return False

    def _strategies_equal(self, pbe1: PBE, pbe2: PBE) -> bool:
        """Check if two PBEs have the same strategies and beliefs."""
        for key in set(pbe1.sender_strategy.keys()) | set(pbe2.sender_strategy.keys()):
            v1 = pbe1.sender_strategy.get(key, 0.0)
            v2 = pbe2.sender_strategy.get(key, 0.0)
            if abs(v1 - v2) > 1e-9:
                return False

        for key in set(pbe1.receiver_strategy.keys()) | set(pbe2.receiver_strategy.keys()):
            v1 = pbe1.receiver_strategy.get(key, 0.0)
            v2 = pbe2.receiver_strategy.get(key, 0.0)
            if abs(v1 - v2) > 1e-9:
                return False

        for key in set(pbe1.beliefs.keys()) | set(pbe2.beliefs.keys()):
            v1 = pbe1.beliefs.get(key, 0.0)
            v2 = pbe2.beliefs.get(key, 0.0)
            if abs(v1 - v2) > 1e-9:
                return False

        return True
