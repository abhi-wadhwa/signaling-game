"""
Spence (1973) job market signaling model.

Setup:
  - Workers have types theta in {theta_L, theta_H} with prior (1-p, p)
  - Workers choose education level e >= 0
  - Education cost: c(e, theta) = e / theta  (lower types pay more per unit)
  - Firms observe e, set wage w(e) = E[theta | e] in competitive equilibrium
  - Worker payoff: w(e) - c(e, theta) = w(e) - e/theta

Equilibrium types:
  - Separating: theta_H picks e* > 0, theta_L picks e=0
    Incentive compatibility: theta_L won't mimic, theta_H prefers signaling
  - Pooling: both types pick same e, wage = E[theta]

The least-cost separating equilibrium has e* s.t. the low type is exactly
indifferent between mimicking and not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpenceEquilibrium:
    """Result of Spence model equilibrium computation."""

    equilibrium_type: str  # "separating" or "pooling"
    education_levels: dict[str, float]  # type_name -> education choice
    wages: dict[str, float]  # type_name -> equilibrium wage (or education -> wage)
    payoffs: dict[str, float]  # type_name -> net payoff
    wage_schedule: list[tuple[float, float]] = field(default_factory=list)
    description: str = ""


@dataclass
class SpenceModel:
    """
    Spence job market signaling model with two worker types.

    Parameters:
        theta_low: productivity of low type
        theta_high: productivity of high type
        prob_high: prior probability of high type
        cost_fn: cost function c(e, theta), defaults to e/theta
    """

    theta_low: float
    theta_high: float
    prob_high: float
    cost_fn: callable = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        assert 0 < self.theta_low < self.theta_high
        assert 0 < self.prob_high < 1
        if self.cost_fn is None:
            self.cost_fn = lambda e, theta: e / theta

    @property
    def prob_low(self) -> float:
        return 1.0 - self.prob_high

    @property
    def expected_theta(self) -> float:
        """Unconditional expected productivity."""
        return self.prob_low * self.theta_low + self.prob_high * self.theta_high

    def cost(self, e: float, theta: float) -> float:
        """Education cost for type theta at education level e."""
        return self.cost_fn(e, theta)

    def separating_equilibrium(self) -> SpenceEquilibrium:
        """
        Compute the least-cost separating equilibrium.

        Low type: e_L = 0, wage = theta_L
        High type: e_H = e* where low type is indifferent about mimicking

        IC for low type: theta_L - 0 >= theta_H - c(e*, theta_L)
        => c(e*, theta_L) >= theta_H - theta_L
        => e* / theta_L >= theta_H - theta_L
        => e* >= theta_L * (theta_H - theta_L)

        Least-cost: e* = theta_L * (theta_H - theta_L)

        We also check IC for high type:
        theta_H - c(e*, theta_H) >= theta_L - 0
        => theta_H - e*/theta_H >= theta_L
        """
        e_low = 0.0
        # Least-cost separating: low type exactly indifferent
        e_star = self._find_separating_education()

        w_low = self.theta_low
        w_high = self.theta_high

        payoff_low = w_low - self.cost(e_low, self.theta_low)
        payoff_high = w_high - self.cost(e_star, self.theta_high)

        # Build wage schedule for plotting
        wage_schedule = self._separating_wage_schedule(e_star)

        # Verify ICs
        low_deviation = w_high - self.cost(e_star, self.theta_low)
        high_deviation = w_low - self.cost(0.0, self.theta_high)

        desc = (
            f"Least-cost separating equilibrium:\n"
            f"  Low type (theta={self.theta_low}): e=0, wage={w_low:.4f}, "
            f"payoff={payoff_low:.4f}\n"
            f"  High type (theta={self.theta_high}): e={e_star:.4f}, "
            f"wage={w_high:.4f}, payoff={payoff_high:.4f}\n"
            f"  Low type IC: {payoff_low:.4f} >= {low_deviation:.4f} "
            f"({'satisfied' if payoff_low >= low_deviation - 1e-9 else 'VIOLATED'})\n"
            f"  High type IC: {payoff_high:.4f} >= {high_deviation:.4f} "
            f"({'satisfied' if payoff_high >= high_deviation - 1e-9 else 'VIOLATED'})"
        )

        return SpenceEquilibrium(
            equilibrium_type="separating",
            education_levels={"low": e_low, "high": e_star},
            wages={"low": w_low, "high": w_high},
            payoffs={"low": payoff_low, "high": payoff_high},
            wage_schedule=wage_schedule,
            description=desc,
        )

    def pooling_equilibrium(self, e_pool: float = 0.0) -> SpenceEquilibrium:
        """
        Compute pooling equilibrium where both types choose e_pool.

        Wage = E[theta] since firms can't distinguish types.
        For this to be an equilibrium, neither type should want to deviate.
        """
        w_pool = self.expected_theta

        payoff_low = w_pool - self.cost(e_pool, self.theta_low)
        payoff_high = w_pool - self.cost(e_pool, self.theta_high)

        # Check if either type wants to deviate
        # Deviation to e=0 gets wage = ? (off-path belief dependent)
        # Under pessimistic beliefs (off-path -> low type), wage = theta_L
        w_deviate = self.theta_low  # worst-case off-path belief
        dev_low = w_deviate - self.cost(0.0, self.theta_low)
        dev_high = w_deviate - self.cost(0.0, self.theta_high)

        wage_schedule = self._pooling_wage_schedule(e_pool)

        desc = (
            f"Pooling equilibrium at e={e_pool:.4f}:\n"
            f"  Pooling wage = E[theta] = {w_pool:.4f}\n"
            f"  Low type: payoff={payoff_low:.4f}, deviation payoff={dev_low:.4f}\n"
            f"  High type: payoff={payoff_high:.4f}, deviation payoff={dev_high:.4f}\n"
            f"  Sustainable: "
            f"{'yes' if payoff_low >= dev_low - 1e-9 and payoff_high >= dev_high - 1e-9 else 'no (under pessimistic off-path beliefs)'}"
        )

        return SpenceEquilibrium(
            equilibrium_type="pooling",
            education_levels={"low": e_pool, "high": e_pool},
            wages={"low": w_pool, "high": w_pool},
            payoffs={"low": payoff_low, "high": payoff_high},
            wage_schedule=wage_schedule,
            description=desc,
        )

    def all_equilibria(self) -> list[SpenceEquilibrium]:
        """Return both separating and pooling equilibria."""
        return [self.separating_equilibrium(), self.pooling_equilibrium()]

    def _find_separating_education(self) -> float:
        """
        Find least-cost separating education level e*.

        For the standard cost c(e, theta) = e/theta:
            e* = theta_L * (theta_H - theta_L)

        For general cost functions, solve numerically:
            c(e*, theta_L) = theta_H - theta_L
        """
        # Try analytical solution for standard cost
        target_cost = self.theta_high - self.theta_low

        # Binary search for general cost functions
        lo, hi = 0.0, 100.0 * self.theta_high
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if self.cost(mid, self.theta_low) < target_cost:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2.0

    def _separating_wage_schedule(
        self, e_star: float
    ) -> list[tuple[float, float]]:
        """Generate wage schedule points for separating equilibrium."""
        points = []
        max_e = e_star * 2.0 + 1.0
        for e in np.linspace(0, max_e, 200):
            if e < e_star - 1e-9:
                w = self.theta_low
            elif e >= e_star - 1e-9:
                w = self.theta_high
            points.append((float(e), float(w)))
        return points

    def _pooling_wage_schedule(
        self, e_pool: float
    ) -> list[tuple[float, float]]:
        """Generate wage schedule points for pooling equilibrium."""
        points = []
        max_e = max(e_pool * 2.0, 2.0)
        for e in np.linspace(0, max_e, 200):
            # On path: all education = e_pool
            if abs(e - e_pool) < max_e / 200:
                w = self.expected_theta
            else:
                # Off-path: pessimistic beliefs => wage = theta_L
                w = self.theta_low
            points.append((float(e), float(w)))
        return points

    def indifference_curves(
        self,
        theta: float,
        target_payoff: float,
        e_range: tuple[float, float] = (0.0, 5.0),
        num_points: int = 200,
    ) -> list[tuple[float, float]]:
        """
        Compute indifference curve: w - c(e, theta) = target_payoff
        => w = target_payoff + c(e, theta)
        """
        points = []
        for e in np.linspace(e_range[0], e_range[1], num_points):
            w = target_payoff + self.cost(float(e), theta)
            points.append((float(e), float(w)))
        return points
