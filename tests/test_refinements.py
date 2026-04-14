"""Tests for equilibrium refinements (Intuitive Criterion and D1)."""

import numpy as np

from src.core.beer_quiche import BEER, QUICHE, TOUGH, WEAK, BeerQuicheGame
from src.core.d1_criterion import d1_criterion_filter
from src.core.intuitive_criterion import (
    check_intuitive_criterion,
    intuitive_criterion_filter,
)
from src.core.pbe_solver import PBESolver
from src.core.signaling import Action, SenderType, Signal, SignalingGame


class TestIntuitiveCriterion:
    """Tests for the Intuitive Criterion (Cho-Kreps)."""

    def test_beer_quiche_selects_beer_pooling(self) -> None:
        """
        The classic result: Intuitive Criterion selects pooling-on-Beer
        when P(Tough) is high (0.9).

        The pooling-on-Quiche equilibrium fails because:
        - Beer is off-path
        - Beer is equilibrium-dominated for Weak (even best response NF gives 2 < 3)
        - Tough would benefit from Beer if receiver believes mu(Tough|Beer)=1
          (gets 3 > 2)
        """
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        pbes = bq.enumerate_pure_pbe()

        survivors = intuitive_criterion_filter(pbes, game)

        # At least one pooling-on-Beer should survive
        beer_pooling_survivors = [
            p for p in survivors
            if p.sender_strategy.get((TOUGH, BEER), 0) > 0.5
            and p.sender_strategy.get((WEAK, BEER), 0) > 0.5
        ]
        assert len(beer_pooling_survivors) >= 1

        # Pooling-on-Quiche should NOT survive
        quiche_pooling_survivors = [
            p for p in survivors
            if p.sender_strategy.get((TOUGH, QUICHE), 0) > 0.5
            and p.sender_strategy.get((WEAK, QUICHE), 0) > 0.5
        ]
        assert len(quiche_pooling_survivors) == 0

    def test_separating_passes_ic(self) -> None:
        """Separating equilibria (if they exist) should pass IC."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        pbes = bq.enumerate_pure_pbe()

        separating = [p for p in pbes if p.equilibrium_type == "separating"]
        for pbe in separating:
            assert check_intuitive_criterion(pbe, game)

    def test_filter_returns_subset(self) -> None:
        """IC filter should return a subset of the original PBE list."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        pbes = bq.enumerate_pure_pbe()
        survivors = intuitive_criterion_filter(pbes, game)

        assert len(survivors) <= len(pbes)

    def test_on_path_signals_not_affected(self) -> None:
        """IC only restricts off-path beliefs, on-path signals are fine."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()

        # A separating equilibrium has no off-path signals
        # So it should always pass IC
        pbes = bq.enumerate_pure_pbe()
        sep = [p for p in pbes if p.equilibrium_type == "separating"]
        for pbe in sep:
            assert check_intuitive_criterion(pbe, game)


class TestD1Criterion:
    """Tests for the D1 criterion."""

    def test_d1_at_least_as_strong_as_ic(self) -> None:
        """D1 survivors should be a subset of IC survivors."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        pbes = bq.enumerate_pure_pbe()

        ic_survivors = intuitive_criterion_filter(pbes, game)
        d1_survivors = d1_criterion_filter(pbes, game)

        # Every D1 survivor should also be an IC survivor
        for pbe in d1_survivors:
            assert pbe in ic_survivors

    def test_d1_selects_beer_pooling(self) -> None:
        """D1 should also select pooling-on-Beer for standard Beer-Quiche."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        pbes = bq.enumerate_pure_pbe()

        survivors = d1_criterion_filter(pbes, game)

        beer_pooling = [
            p for p in survivors
            if p.sender_strategy.get((TOUGH, BEER), 0) > 0.5
            and p.sender_strategy.get((WEAK, BEER), 0) > 0.5
        ]
        assert len(beer_pooling) >= 1

    def test_d1_eliminates_quiche_pooling(self) -> None:
        """D1 should eliminate pooling-on-Quiche."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        pbes = bq.enumerate_pure_pbe()

        survivors = d1_criterion_filter(pbes, game)

        quiche_pooling = [
            p for p in survivors
            if p.sender_strategy.get((TOUGH, QUICHE), 0) > 0.5
            and p.sender_strategy.get((WEAK, QUICHE), 0) > 0.5
        ]
        assert len(quiche_pooling) == 0

    def test_d1_filter_returns_subset(self) -> None:
        """D1 filter should return a subset."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        pbes = bq.enumerate_pure_pbe()
        survivors = d1_criterion_filter(pbes, game)
        assert len(survivors) <= len(pbes)


class TestPBESolver:
    """Tests for the general PBE solver."""

    def test_beer_quiche_finds_pbe(self) -> None:
        """PBE solver should find PBE for Beer-Quiche."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        solver = PBESolver(game)
        pbes = solver.find_pure_pbe()
        assert len(pbes) >= 2  # At least pooling on Beer and Quiche

    def test_custom_game(self) -> None:
        """Test PBE solver on a custom 2x2x2 signaling game."""
        types = [SenderType("A", 0), SenderType("B", 1)]
        signals = [Signal("S1", 0), Signal("S2", 1)]
        actions = [Action("a1", 0), Action("a2", 1)]
        prior = np.array([0.5, 0.5])

        # Simple payoffs where separating is obvious
        sender_payoff = np.array([
            # Type A
            [[2.0, 0.0],   # S1: a1=2, a2=0
             [0.0, 1.0]],  # S2: a1=0, a2=1
            # Type B
            [[0.0, 1.0],   # S1: a1=0, a2=1
             [2.0, 0.0]],  # S2: a1=2, a2=0
        ])

        receiver_payoff = np.array([
            # Type A
            [[1.0, 0.0],   # S1: a1=1, a2=0
             [0.0, 1.0]],  # S2: a1=0, a2=1
            # Type B
            [[0.0, 1.0],   # S1: a1=0, a2=1
             [1.0, 0.0]],  # S2: a1=1, a2=0
        ])

        game = SignalingGame(
            types=types,
            signals=signals,
            actions=actions,
            prior=prior,
            sender_payoff=sender_payoff,
            receiver_payoff=receiver_payoff,
        )

        solver = PBESolver(game)
        pbes = solver.find_pure_pbe()
        assert len(pbes) >= 1

    def test_all_pbe_satisfy_sequential_rationality(self) -> None:
        """Every PBE found should satisfy sequential rationality."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        solver = PBESolver(game)
        pbes = solver.find_pure_pbe()

        for pbe in pbes:
            # Check receiver BR
            for m in range(game.num_signals):
                belief = pbe.get_belief(m)
                if not belief:
                    continue
                # Find the action chosen by receiver
                action_dist = pbe.get_receiver_action(m)
                if not action_dist:
                    continue
                chosen_a = max(action_dist, key=action_dist.get)
                # Check it's a best response
                eu_chosen = game.expected_receiver_payoff(m, chosen_a, belief)
                for a in range(game.num_actions):
                    eu_alt = game.expected_receiver_payoff(m, a, belief)
                    assert eu_chosen >= eu_alt - 1e-9

    def test_bayes_consistency(self) -> None:
        """On-path beliefs should be consistent with Bayes' rule."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        solver = PBESolver(game)
        pbes = solver.find_pure_pbe()

        for pbe in pbes:
            for m in range(game.num_signals):
                if pbe.is_on_path(m):
                    # Bayes update should match stored beliefs
                    bayes_beliefs = game.bayes_update(m, pbe.sender_strategy)
                    for t in range(game.num_types):
                        stored = pbe.beliefs.get((m, t), 0.0)
                        computed = bayes_beliefs.get(t, 0.0)
                        assert abs(stored - computed) < 1e-9, (
                            f"Bayes inconsistency at signal {m}, type {t}: "
                            f"stored={stored}, computed={computed}"
                        )

    def test_pbe_classification(self) -> None:
        """PBE should be correctly classified."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        solver = PBESolver(game)
        pbes = solver.find_pure_pbe()

        for pbe in pbes:
            assert pbe.equilibrium_type in {"separating", "pooling", "semi-separating"}

    def test_find_all_pbe_includes_pure(self) -> None:
        """find_all_pbe should include at least the pure PBE."""
        bq = BeerQuicheGame(prob_tough=0.9)
        game = bq.to_signaling_game()
        solver = PBESolver(game)
        pure = solver.find_pure_pbe()
        all_pbe = solver.find_all_pbe()
        assert len(all_pbe) >= len(pure)
