"""Tests for the Crawford-Sobel cheap talk model."""

import pytest
import numpy as np

from src.core.crawford_sobel import CrawfordSobelModel


class TestCrawfordSobel:
    """Tests for CrawfordSobelModel."""

    def test_basic_construction(self) -> None:
        model = CrawfordSobelModel(bias=0.1)
        assert model.bias == 0.1

    def test_negative_bias_rejected(self) -> None:
        with pytest.raises(AssertionError):
            CrawfordSobelModel(bias=-0.1)

    def test_max_partitions_formula(self) -> None:
        """
        N*(b) = floor((1 + sqrt(1 + 2/b)) / 2)

        For b=0.1: N* = floor((1 + sqrt(21)) / 2) = floor(2.79) = 2
        Wait, let's compute properly:
        sqrt(1 + 2/0.1) = sqrt(21) = 4.583
        (1 + 4.583) / 2 = 2.79
        But we need to verify: N*(N-1)*2*b < 1
        N=2: 2*1*0.2 = 0.4 < 1. OK.
        N=3: 3*2*0.2 = 1.2 > 1. Fail.
        So N*(0.1) = 2.
        """
        model = CrawfordSobelModel(bias=0.1)
        assert model.max_partitions() == 2

    def test_max_partitions_small_bias(self) -> None:
        """For b=0.01: N*(N-1)*0.02 < 1 => N*(N-1) < 50 => N*=7."""
        model = CrawfordSobelModel(bias=0.01)
        n_star = model.max_partitions()
        # N=7: 7*6*0.02=0.84 < 1. N=8: 8*7*0.02=1.12 > 1.
        assert n_star == 7

    def test_max_partitions_large_bias(self) -> None:
        """For b=0.5: 1*0*1.0=0 < 1, but N=2: 2*1*1.0=2 > 1. So N*=1."""
        model = CrawfordSobelModel(bias=0.5)
        assert model.max_partitions() == 1

    def test_partition_count_matches_nstar(self) -> None:
        """Number of equilibria found should match N*."""
        for bias in [0.05, 0.1, 0.15, 0.2]:
            model = CrawfordSobelModel(bias=bias)
            n_star = model.max_partitions()
            equilibria = model.all_partition_equilibria()
            assert len(equilibria) == n_star, (
                f"bias={bias}: expected {n_star} equilibria, got {len(equilibria)}"
            )

    def test_babbling_always_exists(self) -> None:
        """N=1 babbling equilibrium should always exist."""
        for bias in [0.01, 0.1, 0.5, 1.0]:
            model = CrawfordSobelModel(bias=bias)
            eq = model.babbling_equilibrium()
            assert eq.num_partitions == 1
            assert len(eq.boundaries) == 2
            assert eq.boundaries[0] == 0.0
            assert eq.boundaries[1] == 1.0
            assert len(eq.actions) == 1
            assert abs(eq.actions[0] - 0.5) < 1e-9

    def test_babbling_receiver_eu(self) -> None:
        """
        Babbling: action y=0.5 always.
        E[u_R] = -E[(0.5 - theta)^2] = -integral_0^1 (0.5 - t)^2 dt = -1/12
        """
        model = CrawfordSobelModel(bias=0.1)
        eq = model.babbling_equilibrium()
        assert abs(eq.receiver_eu - (-1.0 / 12.0)) < 1e-6

    def test_boundaries_valid(self) -> None:
        """Boundaries should be strictly increasing from 0 to 1."""
        model = CrawfordSobelModel(bias=0.05)
        for n in range(1, model.max_partitions() + 1):
            eq = model.partition_equilibrium(n)
            assert eq is not None
            assert eq.boundaries[0] == 0.0
            assert abs(eq.boundaries[-1] - 1.0) < 1e-9
            for i in range(len(eq.boundaries) - 1):
                assert eq.boundaries[i] < eq.boundaries[i + 1]

    def test_actions_are_interval_midpoints(self) -> None:
        """Receiver actions should be the midpoints of intervals."""
        model = CrawfordSobelModel(bias=0.05)
        for n in range(1, model.max_partitions() + 1):
            eq = model.partition_equilibrium(n)
            assert eq is not None
            for i in range(n):
                midpoint = (eq.boundaries[i] + eq.boundaries[i + 1]) / 2.0
                assert abs(eq.actions[i] - midpoint) < 1e-9

    def test_more_partitions_better_for_receiver(self) -> None:
        """More partitions = more information = better for receiver."""
        model = CrawfordSobelModel(bias=0.05)
        equilibria = model.all_partition_equilibria()
        for i in range(len(equilibria) - 1):
            assert equilibria[i + 1].receiver_eu > equilibria[i].receiver_eu - 1e-9

    def test_two_partition_boundaries(self) -> None:
        """
        For N=2, b=0.1:
        a_1 = (1 - 2*1*0.2) / 2 = (1 - 0.4) / 2 = 0.3
        Boundaries: [0, 0.3, 1]
        """
        model = CrawfordSobelModel(bias=0.1)
        eq = model.partition_equilibrium(2)
        assert eq is not None
        assert abs(eq.boundaries[1] - 0.3) < 1e-6

    def test_invalid_n_returns_none(self) -> None:
        """Requesting too many partitions returns None."""
        model = CrawfordSobelModel(bias=0.1)
        eq = model.partition_equilibrium(10)
        assert eq is None

    def test_most_informative(self) -> None:
        """Most informative should have max partitions."""
        model = CrawfordSobelModel(bias=0.05)
        eq = model.most_informative_equilibrium()
        assert eq.num_partitions == model.max_partitions()

    def test_information_loss_babbling(self) -> None:
        """Babbling should have information_loss = 1 (ratio relative to babbling)."""
        model = CrawfordSobelModel(bias=0.1)
        loss = model.information_loss(1)
        assert abs(loss - 1.0) < 1e-9

    def test_sender_eu_babbling(self) -> None:
        """
        Babbling with b=0.1:
        E[u_S] = -E[(0.5 - theta - 0.1)^2] = -E[(0.4 - theta)^2]
        = -(0.16 - 0.8*0.5 + 1/3) = -(0.16 - 0.4 + 0.333) = -0.0933
        """
        model = CrawfordSobelModel(bias=0.1)
        eq = model.babbling_equilibrium()
        # -(0.4^2 - 2*0.4*0.5 + 1/3) = -(0.16 - 0.4 + 0.333) = -0.0933
        expected = -(0.4**2 - 2 * 0.4 * 0.5 + 1.0 / 3.0)
        assert abs(eq.sender_eu - expected) < 1e-4
