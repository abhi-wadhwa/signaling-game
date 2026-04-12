"""
Crawford-Sobel (1982) cheap talk model.

Setup:
  - Sender observes state theta ~ U[0,1]
  - Sender sends costless message m (cheap talk)
  - Receiver takes action y
  - Sender payoff: u_S(y, theta, b) = -(y - theta - b)^2
  - Receiver payoff: u_R(y, theta) = -(y - theta)^2
  - b >= 0 is the sender's bias

Key result: In equilibrium, the state space [0,1] is partitioned into
N intervals, with the sender reporting which interval theta belongs to.
The receiver takes the conditional mean action within each interval.

Maximum number of credible partitions N*(b):
  The largest N such that a partition equilibrium exists.
  Determined by the condition that boundary indifference holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PartitionEquilibrium:
    """A partition equilibrium of the Crawford-Sobel model."""

    num_partitions: int
    boundaries: list[float]  # [0, a_1, a_2, ..., a_{N-1}, 1]
    actions: list[float]  # receiver's action in each interval
    sender_eu: float  # sender's ex ante expected utility
    receiver_eu: float  # receiver's ex ante expected utility
    description: str = ""


@dataclass
class CrawfordSobelModel:
    """
    Crawford-Sobel cheap talk model.

    Parameters:
        bias: sender's bias parameter b >= 0
    """

    bias: float

    def __post_init__(self) -> None:
        assert self.bias >= 0, "Bias must be non-negative"

    def max_partitions(self) -> int:
        """
        Compute the maximum number of partition elements N*(b).

        N*(b) is the largest integer N such that a partition equilibrium
        with N elements exists.

        For uniform-quadratic case:
            N*(b) is the largest N such that:
            a_i boundaries can be constructed with all intervals positive.

        The formula: N*(b) is the largest N with N*(N-1)*2*b < 1,
        equivalently N < (1 + sqrt(1 + 2/b)) / 2 for b > 0.
        """
        if self.bias == 0:
            return float("inf")  # type: ignore[return-value]

        # N*(b) = floor((1 + sqrt(1 + 2/b)) / 2)
        n_star = int((1 + np.sqrt(1 + 2.0 / self.bias)) / 2.0)

        # Verify by checking boundary construction
        while n_star > 0:
            boundaries = self._compute_boundaries(n_star)
            if boundaries is not None:
                return n_star
            n_star -= 1

        return 1  # babbling always exists

    def _compute_boundaries(self, n: int) -> list[float] | None:
        """
        Compute partition boundaries for an N-element equilibrium.

        Boundary condition (indifference): for uniform-quadratic,
            a_i = a_{i-1} + (a_{i+1} - a_{i-1})/2  is NOT correct.

        The correct recursion:
            a_0 = 0
            a_N = 1
            Indifference of boundary type a_i:
                (y_i - a_i - b)^2 = (y_{i+1} - a_i - b)^2
            where y_i = (a_{i-1} + a_i)/2 is receiver's action in interval i.

            This gives: a_{i+1} = 2*a_i - a_{i-1} + 4*b
            (the uniform-quadratic recursion).
        """
        if n == 1:
            return [0.0, 1.0]

        # Forward recursion: a_0=0, a_1=x (free parameter), a_{i+1} = 2*a_i - a_{i-1} + 4b
        # We need a_N = 1, so solve for x = a_1.
        # Analytically: a_i = i*x + i*(i-1)*2*b for the uniform-quadratic case.
        # Set a_N = 1: N*x + N*(N-1)*2*b = 1
        # => x = (1 - N*(N-1)*2*b) / N

        x = (1.0 - n * (n - 1) * 2.0 * self.bias) / n

        if x <= 1e-12:
            return None  # No valid equilibrium with N partitions

        boundaries = [0.0]
        for i in range(1, n + 1):
            a_i = i * x + i * (i - 1) * 2.0 * self.bias
            if i < n and (a_i <= boundaries[-1] + 1e-12 or a_i >= 1.0 - 1e-12):
                return None
            boundaries.append(a_i)

        # Fix floating point: last boundary should be exactly 1
        if abs(boundaries[-1] - 1.0) > 1e-6:
            return None
        boundaries[-1] = 1.0

        return boundaries

    def partition_equilibrium(self, n: int) -> PartitionEquilibrium | None:
        """
        Compute the N-element partition equilibrium.

        Returns None if no equilibrium with N partitions exists.
        """
        boundaries = self._compute_boundaries(n)
        if boundaries is None:
            return None

        # Receiver actions: conditional mean in each interval
        actions = []
        for i in range(n):
            y_i = (boundaries[i] + boundaries[i + 1]) / 2.0
            actions.append(y_i)

        # Expected utilities
        sender_eu = self._compute_sender_eu(boundaries, actions)
        receiver_eu = self._compute_receiver_eu(boundaries, actions)

        desc = (
            f"{n}-partition equilibrium (b={self.bias}):\n"
            f"  Boundaries: {[f'{b:.4f}' for b in boundaries]}\n"
            f"  Actions: {[f'{a:.4f}' for a in actions]}\n"
            f"  Sender E[u]: {sender_eu:.6f}\n"
            f"  Receiver E[u]: {receiver_eu:.6f}"
        )

        return PartitionEquilibrium(
            num_partitions=n,
            boundaries=boundaries,
            actions=actions,
            sender_eu=sender_eu,
            receiver_eu=receiver_eu,
            description=desc,
        )

    def all_partition_equilibria(self) -> list[PartitionEquilibrium]:
        """Compute all partition equilibria from N=1 (babbling) up to N*."""
        n_max = self.max_partitions()
        if isinstance(n_max, float):  # infinity case
            n_max = 50  # cap for computation

        equilibria = []
        for n in range(1, n_max + 1):
            eq = self.partition_equilibrium(n)
            if eq is not None:
                equilibria.append(eq)
        return equilibria

    def babbling_equilibrium(self) -> PartitionEquilibrium:
        """The babbling (uninformative) equilibrium: N=1."""
        eq = self.partition_equilibrium(1)
        assert eq is not None
        return eq

    def most_informative_equilibrium(self) -> PartitionEquilibrium:
        """The equilibrium with the maximum number of partitions."""
        n_max = self.max_partitions()
        if isinstance(n_max, float):
            n_max = 50
        eq = self.partition_equilibrium(n_max)
        assert eq is not None
        return eq

    def _compute_sender_eu(
        self, boundaries: list[float], actions: list[float]
    ) -> float:
        """
        Sender's ex ante expected utility.
        E[u_S] = -E[(y - theta - b)^2]

        For interval [a_{i-1}, a_i] with action y_i:
        integral_{a_{i-1}}^{a_i} -(y_i - theta - b)^2 dtheta
        """
        total = 0.0
        n = len(actions)
        for i in range(n):
            lo = boundaries[i]
            hi = boundaries[i + 1]
            y = actions[i]
            # integral of -(y - theta - b)^2 from lo to hi
            # = -[y-b)^2*(hi-lo) - (y-b)*(hi^2 - lo^2) + (hi^3 - lo^3)/3]
            # Let c = y - b
            c = y - self.bias
            integral = -(
                c * c * (hi - lo)
                - 2.0 * c * (hi * hi - lo * lo) / 2.0
                + (hi**3 - lo**3) / 3.0
            )
            total += integral
        return total

    def _compute_receiver_eu(
        self, boundaries: list[float], actions: list[float]
    ) -> float:
        """
        Receiver's ex ante expected utility.
        E[u_R] = -E[(y - theta)^2]
        """
        total = 0.0
        n = len(actions)
        for i in range(n):
            lo = boundaries[i]
            hi = boundaries[i + 1]
            y = actions[i]
            # integral of -(y - theta)^2 from lo to hi
            integral = -(
                y * y * (hi - lo)
                - 2.0 * y * (hi * hi - lo * lo) / 2.0
                + (hi**3 - lo**3) / 3.0
            )
            total += integral
        return total

    def information_loss(self, n: int) -> float:
        """
        Fraction of information lost relative to full revelation.

        Full revelation: E[u_R] = 0 (action = theta always)
        Babbling: E[u_R] = -1/12 (action = 1/2 always)
        """
        eq = self.partition_equilibrium(n)
        if eq is None:
            return 1.0
        babbling = self.babbling_equilibrium()
        if babbling.receiver_eu == 0:
            return 0.0
        return eq.receiver_eu / babbling.receiver_eu
