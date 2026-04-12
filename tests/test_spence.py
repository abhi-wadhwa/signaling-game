"""Tests for the Spence job market signaling model."""

import pytest

from src.core.spence import SpenceModel


class TestSpenceModel:
    """Tests for SpenceModel."""

    def test_basic_construction(self) -> None:
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        assert model.theta_low == 1.0
        assert model.theta_high == 2.0
        assert model.prob_high == 0.5
        assert model.prob_low == 0.5

    def test_expected_theta(self) -> None:
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        assert abs(model.expected_theta - 1.5) < 1e-9

    def test_expected_theta_asymmetric(self) -> None:
        model = SpenceModel(theta_low=1.0, theta_high=3.0, prob_high=0.25)
        expected = 0.75 * 1.0 + 0.25 * 3.0  # 1.5
        assert abs(model.expected_theta - expected) < 1e-9

    def test_separating_wages_equal_marginal_products(self) -> None:
        """In separating equilibrium, wages should equal marginal products (theta)."""
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        eq = model.separating_equilibrium()

        assert eq.equilibrium_type == "separating"
        assert abs(eq.wages["low"] - model.theta_low) < 1e-9
        assert abs(eq.wages["high"] - model.theta_high) < 1e-9

    def test_separating_wages_different_params(self) -> None:
        """Wages = marginal products for various parameter settings."""
        for tl, th, p in [(0.5, 1.5, 0.3), (1.0, 3.0, 0.7), (2.0, 5.0, 0.1)]:
            model = SpenceModel(theta_low=tl, theta_high=th, prob_high=p)
            eq = model.separating_equilibrium()
            assert abs(eq.wages["low"] - tl) < 1e-9
            assert abs(eq.wages["high"] - th) < 1e-9

    def test_separating_low_type_education_zero(self) -> None:
        """Low type chooses e=0 in separating equilibrium."""
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        eq = model.separating_equilibrium()
        assert abs(eq.education_levels["low"]) < 1e-9

    def test_separating_education_level(self) -> None:
        """
        Least-cost separating: e* = theta_L * (theta_H - theta_L).
        For theta_L=1, theta_H=2: e* = 1 * 1 = 1.
        """
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        eq = model.separating_equilibrium()
        expected_e = 1.0 * (2.0 - 1.0)  # = 1.0
        assert abs(eq.education_levels["high"] - expected_e) < 0.01

    def test_separating_ic_low_type(self) -> None:
        """Low type should not want to mimic high type."""
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        eq = model.separating_equilibrium()
        # Low type equilibrium payoff
        u_low = eq.payoffs["low"]
        # If low type mimics: wage=theta_H, cost=e*/theta_L
        e_star = eq.education_levels["high"]
        u_mimic = model.theta_high - model.cost(e_star, model.theta_low)
        assert u_low >= u_mimic - 1e-9

    def test_separating_ic_high_type(self) -> None:
        """High type should prefer signaling to being treated as low."""
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        eq = model.separating_equilibrium()
        u_high = eq.payoffs["high"]
        # If high type deviates to e=0: wage=theta_L
        u_deviate = model.theta_low
        assert u_high >= u_deviate - 1e-9

    def test_pooling_equilibrium_wage(self) -> None:
        """Pooling wage = E[theta]."""
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        eq = model.pooling_equilibrium()
        assert eq.equilibrium_type == "pooling"
        assert abs(eq.wages["low"] - model.expected_theta) < 1e-9
        assert abs(eq.wages["high"] - model.expected_theta) < 1e-9

    def test_all_equilibria_returns_both(self) -> None:
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        eqs = model.all_equilibria()
        assert len(eqs) == 2
        types = {eq.equilibrium_type for eq in eqs}
        assert types == {"separating", "pooling"}

    def test_indifference_curves(self) -> None:
        """Indifference curves should pass through the target payoff at e=0."""
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        ic = model.indifference_curves(1.0, 1.0, (0, 3))
        # At e=0: w = payoff + c(0,1) = 1.0 + 0 = 1.0
        assert abs(ic[0][1] - 1.0) < 1e-9

    def test_cost_function(self) -> None:
        """Default cost c(e, theta) = e / theta."""
        model = SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=0.5)
        assert abs(model.cost(2.0, 1.0) - 2.0) < 1e-9
        assert abs(model.cost(2.0, 2.0) - 1.0) < 1e-9

    def test_custom_cost_function(self) -> None:
        """Model with custom cost function."""
        model = SpenceModel(
            theta_low=1.0,
            theta_high=2.0,
            prob_high=0.5,
            cost_fn=lambda e, theta: e**2 / theta,
        )
        assert abs(model.cost(2.0, 1.0) - 4.0) < 1e-9

    def test_invalid_params(self) -> None:
        with pytest.raises(AssertionError):
            SpenceModel(theta_low=2.0, theta_high=1.0, prob_high=0.5)

    def test_invalid_prob(self) -> None:
        with pytest.raises(AssertionError):
            SpenceModel(theta_low=1.0, theta_high=2.0, prob_high=1.5)
