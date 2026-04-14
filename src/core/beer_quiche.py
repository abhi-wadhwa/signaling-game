"""
Beer-Quiche signaling game (Cho & Kreps, 1987).

Setup:
  - Two sender types: Tough (probability p) and Weak (probability 1-p)
  - Two signals: Beer (B) or Quiche (Q)
  - Two receiver actions: Fight (F) or Not Fight (NF)

Standard payoffs:
  - Tough type prefers Beer, Weak type prefers Quiche (breakfast preference)
  - Both types prefer Not Fight over Fight
  - Receiver prefers to Fight Weak, Not Fight Tough

Payoff matrix (Tough type):
  Beer+NF: 3, Beer+F: 1, Quiche+NF: 2, Quiche+F: 0

Payoff matrix (Weak type):
  Beer+NF: 2, Beer+F: 0, Quiche+NF: 3, Quiche+F: 1

Receiver payoffs:
  Fight Tough: 0, NF Tough: 1
  Fight Weak: 1, NF Weak: 0

With p = 0.9:
  - Pooling on Beer: PBE (both types choose Beer)
  - Pooling on Quiche: PBE (both types choose Quiche)
  - Intuitive Criterion selects pooling on Beer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.signaling import PBE, Action, SenderType, Signal, SignalingGame

# Standard type/signal/action indices
TOUGH, WEAK = 0, 1
BEER, QUICHE = 0, 1
FIGHT, NOT_FIGHT = 0, 1


@dataclass
class BeerQuicheGame:
    """
    The Beer-Quiche signaling game.

    Parameters:
        prob_tough: prior probability of Tough type (default 0.9)
        sender_payoffs: 3D array [type][signal][action] for sender
        receiver_payoffs: 3D array [type][signal][action] for receiver
    """

    prob_tough: float = 0.9
    sender_payoffs: np.ndarray | None = None
    receiver_payoffs: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.sender_payoffs is None:
            # Standard Beer-Quiche payoffs
            # [type][signal][action] where action: 0=Fight, 1=NotFight
            self.sender_payoffs = np.array([
                # Tough
                [[1.0, 3.0],   # Beer: Fight=1, NF=3
                 [0.0, 2.0]],  # Quiche: Fight=0, NF=2
                # Weak
                [[0.0, 2.0],   # Beer: Fight=0, NF=2
                 [1.0, 3.0]],  # Quiche: Fight=1, NF=3
            ])

        if self.receiver_payoffs is None:
            # Receiver wants to fight Weak, not fight Tough
            self.receiver_payoffs = np.array([
                # Tough
                [[0.0, 1.0],   # Beer: Fight=0, NF=1
                 [0.0, 1.0]],  # Quiche: Fight=0, NF=1
                # Weak
                [[1.0, 0.0],   # Beer: Fight=1, NF=0
                 [1.0, 0.0]],  # Quiche: Fight=1, NF=0
            ])

    def to_signaling_game(self) -> SignalingGame:
        """Convert to the general SignalingGame representation."""
        types = [SenderType("Tough", TOUGH), SenderType("Weak", WEAK)]
        signals = [Signal("Beer", BEER), Signal("Quiche", QUICHE)]
        actions = [Action("Fight", FIGHT), Action("Not Fight", NOT_FIGHT)]
        prior = np.array([self.prob_tough, 1.0 - self.prob_tough])

        return SignalingGame(
            types=types,
            signals=signals,
            actions=actions,
            prior=prior,
            sender_payoff=self.sender_payoffs,
            receiver_payoff=self.receiver_payoffs,
        )

    def enumerate_pure_pbe(self) -> list[PBE]:
        """
        Enumerate all pure-strategy PBE of the Beer-Quiche game.

        Strategy profiles to check:
        Sender: (Tough's signal, Weak's signal) in {B,Q}x{B,Q}
        For each, compute on-path beliefs via Bayes, check receiver BR,
        then check sender optimality. For off-path beliefs, try all extremes.
        """
        game = self.to_signaling_game()
        pbe_list = []

        # All 4 pure sender strategies
        for tough_signal in [BEER, QUICHE]:
            for weak_signal in [BEER, QUICHE]:
                pbes = self._check_pure_strategy(
                    game, tough_signal, weak_signal
                )
                pbe_list.extend(pbes)

        return pbe_list

    def enumerate_all_pbe(self) -> list[PBE]:
        """
        Enumerate all PBE including mixed strategies.

        For the Beer-Quiche game, in addition to pure PBE we check:
        - Semi-separating: one type mixes, the other plays pure
        """
        pbe_list = self.enumerate_pure_pbe()

        # Check semi-separating equilibria
        game = self.to_signaling_game()
        semi_sep = self._check_semi_separating(game)
        pbe_list.extend(semi_sep)

        return pbe_list

    def _check_pure_strategy(
        self,
        game: SignalingGame,
        tough_signal: int,
        weak_signal: int,
    ) -> list[PBE]:
        """Check if a pure sender strategy can be part of a PBE."""
        results = []

        # Build sender strategy
        sender_strat = {
            (TOUGH, BEER): 1.0 if tough_signal == BEER else 0.0,
            (TOUGH, QUICHE): 1.0 if tough_signal == QUICHE else 0.0,
            (WEAK, BEER): 1.0 if weak_signal == BEER else 0.0,
            (WEAK, QUICHE): 1.0 if weak_signal == QUICHE else 0.0,
        }

        # Compute on-path beliefs
        on_path_signals = set()
        if tough_signal == weak_signal:
            # Pooling
            on_path_signals.add(tough_signal)
        else:
            # Separating
            on_path_signals = {tough_signal, weak_signal}

        off_path_signals = {BEER, QUICHE} - on_path_signals

        # For on-path signals, beliefs from Bayes' rule
        on_path_beliefs = {}
        for m in on_path_signals:
            beliefs = game.bayes_update(m, sender_strat)
            on_path_beliefs[m] = beliefs

        # For off-path signals, try all extreme beliefs (mu_tough in {0, 1})
        off_path_belief_options = []
        if off_path_signals:
            # Try mu(Tough | off-path) in {0.0, 1.0}
            off_path_belief_options = [0.0, 1.0]
        else:
            off_path_belief_options = [None]

        for off_belief_tough in off_path_belief_options:
            # Build complete belief system
            beliefs = {}
            for m in on_path_signals:
                for t, prob in on_path_beliefs[m].items():
                    beliefs[(m, t)] = prob

            for m in off_path_signals:
                if off_belief_tough is not None:
                    beliefs[(m, TOUGH)] = off_belief_tough
                    beliefs[(m, WEAK)] = 1.0 - off_belief_tough

            # Compute receiver best responses
            receiver_strat = {}
            for m in [BEER, QUICHE]:
                belief_m = {
                    t: beliefs.get((m, t), 0.0) for t in [TOUGH, WEAK]
                }
                # Expected payoff from each action
                eu_fight = sum(
                    belief_m.get(t, 0.0) * game.u_r(t, m, FIGHT)
                    for t in [TOUGH, WEAK]
                )
                eu_nf = sum(
                    belief_m.get(t, 0.0) * game.u_r(t, m, NOT_FIGHT)
                    for t in [TOUGH, WEAK]
                )

                if eu_fight > eu_nf + 1e-9:
                    receiver_strat[(m, FIGHT)] = 1.0
                    receiver_strat[(m, NOT_FIGHT)] = 0.0
                elif eu_nf > eu_fight + 1e-9:
                    receiver_strat[(m, FIGHT)] = 0.0
                    receiver_strat[(m, NOT_FIGHT)] = 1.0
                else:
                    # Indifferent -- try both pure actions
                    # For simplicity, use Fight (we also handle NF below)
                    receiver_strat[(m, FIGHT)] = 1.0
                    receiver_strat[(m, NOT_FIGHT)] = 0.0

            # Check sender optimality
            sender_ok = True
            for t, chosen_m in [(TOUGH, tough_signal), (WEAK, weak_signal)]:
                other_m = 1 - chosen_m
                # Payoff from chosen signal
                u_chosen = sum(
                    receiver_strat.get((chosen_m, a), 0.0) * game.u_s(t, chosen_m, a)
                    for a in [FIGHT, NOT_FIGHT]
                )
                # Payoff from deviation
                u_deviate = sum(
                    receiver_strat.get((other_m, a), 0.0) * game.u_s(t, other_m, a)
                    for a in [FIGHT, NOT_FIGHT]
                )
                if u_deviate > u_chosen + 1e-9:
                    sender_ok = False
                    break

            if sender_ok:
                # Compute payoffs
                sender_payoffs = {}
                for t, chosen_m in [(TOUGH, tough_signal), (WEAK, weak_signal)]:
                    sender_payoffs[t] = sum(
                        receiver_strat.get((chosen_m, a), 0.0) * game.u_s(t, chosen_m, a)
                        for a in [FIGHT, NOT_FIGHT]
                    )

                signal_names = {BEER: "Beer", QUICHE: "Quiche"}
                pbe = PBE(
                    sender_strategy=sender_strat,
                    receiver_strategy=receiver_strat,
                    beliefs=beliefs,
                    sender_payoffs=sender_payoffs,
                    label=(
                        f"Tough->{signal_names[tough_signal]}, "
                        f"Weak->{signal_names[weak_signal]}"
                    ),
                )
                pbe.classify(2, 2)
                results.append(pbe)

        return results

    def _check_semi_separating(self, game: SignalingGame) -> list[PBE]:
        """
        Check for semi-separating (mixed strategy) PBE.

        One type mixes while the other plays pure.
        """
        results = []

        # Try: Tough plays Beer pure, Weak mixes Beer/Quiche
        for pure_type, pure_signal, mix_type in [
            (TOUGH, BEER, WEAK),
            (TOUGH, QUICHE, WEAK),
            (WEAK, BEER, TOUGH),
            (WEAK, QUICHE, TOUGH),
        ]:
            pbe = self._solve_semi_sep(game, pure_type, pure_signal, mix_type)
            if pbe is not None:
                results.append(pbe)

        return results

    def _solve_semi_sep(
        self,
        game: SignalingGame,
        pure_type: int,
        pure_signal: int,
        mix_type: int,
    ) -> PBE | None:
        """
        Solve for semi-separating equilibrium where pure_type plays pure_signal
        and mix_type mixes.

        For mix_type to mix, they must be indifferent between Beer and Quiche.
        This requires the receiver to mix at one of the signals.
        """
        other_signal = 1 - pure_signal
        p_tough = self.prob_tough
        p_weak = 1.0 - p_tough

        # mix_type sends pure_signal with probability q, other_signal with (1-q)
        # On-path: both signals are on path
        # Beliefs at pure_signal: Bayes update with both types sending it
        # Beliefs at other_signal: only mix_type sends it

        # At other_signal, belief is mu(mix_type | other_signal) = 1.0
        # So receiver knows the type at other_signal
        belief_other = {mix_type: 1.0, pure_type: 0.0}

        # Receiver BR at other_signal
        eu_fight_other = sum(
            belief_other.get(t, 0.0) * game.u_r(t, other_signal, FIGHT)
            for t in [TOUGH, WEAK]
        )
        eu_nf_other = sum(
            belief_other.get(t, 0.0) * game.u_r(t, other_signal, NOT_FIGHT)
            for t in [TOUGH, WEAK]
        )

        # Determine receiver action at other_signal
        if eu_fight_other > eu_nf_other + 1e-9:
            r_other_fight = 1.0
        elif eu_nf_other > eu_fight_other + 1e-9:
            r_other_fight = 0.0
        else:
            r_other_fight = 0.5  # indifferent

        # mix_type payoff at other_signal
        u_mix_other = (
            r_other_fight * game.u_s(mix_type, other_signal, FIGHT)
            + (1 - r_other_fight) * game.u_s(mix_type, other_signal, NOT_FIGHT)
        )

        # For mix_type to mix, need payoff at pure_signal to equal u_mix_other
        # At pure_signal, receiver may need to mix Fight/NF
        # u_mix_pure = r_fight * u_S(mix, pure, F) + (1-r_fight) * u_S(mix, pure, NF)
        u_mix_f = game.u_s(mix_type, pure_signal, FIGHT)
        u_mix_nf = game.u_s(mix_type, pure_signal, NOT_FIGHT)

        if abs(u_mix_nf - u_mix_f) < 1e-12:
            return None  # Can't create indifference by mixing

        # r_fight s.t.: r_fight * u_mix_f + (1-r_fight) * u_mix_nf = u_mix_other
        r_fight_pure = (u_mix_other - u_mix_nf) / (u_mix_f - u_mix_nf)

        if r_fight_pure < -1e-9 or r_fight_pure > 1.0 + 1e-9:
            return None
        r_fight_pure = max(0.0, min(1.0, r_fight_pure))

        # For receiver to mix at pure_signal, they must be indifferent
        # This pins down beliefs at pure_signal
        # eu_fight = eu_nf at pure_signal
        # sum_t mu(t|m) * u_R(t, m, F) = sum_t mu(t|m) * u_R(t, m, NF)
        # 2 types: mu*u_R(0,m,F) + (1-mu)*u_R(1,m,F) = mu*u_R(0,m,NF) + (1-mu)*u_R(1,m,NF)
        # mu = belief that type=TOUGH

        r_tough_f = game.u_r(TOUGH, pure_signal, FIGHT)
        r_tough_nf = game.u_r(TOUGH, pure_signal, NOT_FIGHT)
        r_weak_f = game.u_r(WEAK, pure_signal, FIGHT)
        r_weak_nf = game.u_r(WEAK, pure_signal, NOT_FIGHT)

        denom = (r_tough_f - r_tough_nf) - (r_weak_f - r_weak_nf)
        if abs(denom) < 1e-12:
            return None

        mu_tough_pure = (r_weak_nf - r_weak_f) / denom

        if mu_tough_pure < -1e-9 or mu_tough_pure > 1.0 + 1e-9:
            return None
        mu_tough_pure = max(0.0, min(1.0, mu_tough_pure))

        # From Bayes: mu(TOUGH | pure_signal)
        # = p(TOUGH) * sigma(pure_signal | TOUGH) / [p(T)*sigma(m|T) + p(W)*sigma(m|W)]
        # pure_type always sends pure_signal (prob 1)
        # mix_type sends pure_signal with prob q

        # If pure_type = TOUGH:
        #   mu_tough = p_tough * 1 / (p_tough * 1 + p_weak * q)
        # If pure_type = WEAK:
        #   mu_tough = p_tough * q / (p_tough * q + p_weak * 1)

        if pure_type == TOUGH:
            # mu_tough_pure = p_tough / (p_tough + p_weak * q)
            # => p_tough + p_weak * q = p_tough / mu_tough_pure
            if mu_tough_pure < 1e-12:
                return None
            q = (p_tough / mu_tough_pure - p_tough) / p_weak
        else:
            # pure_type = WEAK, mix_type = TOUGH
            # mu_tough_pure = p_tough * q / (p_tough * q + p_weak)
            if mu_tough_pure < 1e-12:
                return None  # need mu > 0 which needs q > 0
            # p_tough * q + p_weak = p_tough * q / mu_tough_pure
            # p_weak = p_tough * q * (1/mu_tough_pure - 1)
            factor = 1.0 / mu_tough_pure - 1.0
            if abs(p_tough * factor) < 1e-12:
                return None
            q = p_weak / (p_tough * factor)

        if q < -1e-9 or q > 1.0 + 1e-9:
            return None
        q = max(0.0, min(1.0, q))

        # Check pure_type doesn't want to deviate
        u_pure_chosen = (
            r_fight_pure * game.u_s(pure_type, pure_signal, FIGHT)
            + (1 - r_fight_pure) * game.u_s(pure_type, pure_signal, NOT_FIGHT)
        )
        u_pure_deviate = (
            r_other_fight * game.u_s(pure_type, other_signal, FIGHT)
            + (1 - r_other_fight) * game.u_s(pure_type, other_signal, NOT_FIGHT)
        )

        if u_pure_deviate > u_pure_chosen + 1e-9:
            return None

        # Build PBE
        sender_strat = {}
        sender_strat[(pure_type, pure_signal)] = 1.0
        sender_strat[(pure_type, other_signal)] = 0.0
        sender_strat[(mix_type, pure_signal)] = q
        sender_strat[(mix_type, other_signal)] = 1.0 - q

        receiver_strat = {
            (pure_signal, FIGHT): r_fight_pure,
            (pure_signal, NOT_FIGHT): 1.0 - r_fight_pure,
            (other_signal, FIGHT): r_other_fight,
            (other_signal, NOT_FIGHT): 1.0 - r_other_fight,
        }

        beliefs = {
            (pure_signal, TOUGH): mu_tough_pure,
            (pure_signal, WEAK): 1.0 - mu_tough_pure,
            (other_signal, TOUGH): 1.0 if mix_type == TOUGH else 0.0,
            (other_signal, WEAK): 1.0 if mix_type == WEAK else 0.0,
        }

        type_names = {TOUGH: "Tough", WEAK: "Weak"}
        signal_names = {BEER: "Beer", QUICHE: "Quiche"}

        sender_payoffs = {}
        for t in [TOUGH, WEAK]:
            eu = 0.0
            for m in [BEER, QUICHE]:
                sigma_m = sender_strat.get((t, m), 0.0)
                for a in [FIGHT, NOT_FIGHT]:
                    eu += sigma_m * receiver_strat.get((m, a), 0.0) * game.u_s(t, m, a)
            sender_payoffs[t] = eu

        label = (
            f"{type_names[pure_type]}->{signal_names[pure_signal]} (pure), "
            f"{type_names[mix_type]} mixes (q={q:.4f})"
        )

        pbe = PBE(
            sender_strategy=sender_strat,
            receiver_strategy=receiver_strat,
            beliefs=beliefs,
            sender_payoffs=sender_payoffs,
            label=label,
            equilibrium_type="semi-separating",
        )

        return pbe
