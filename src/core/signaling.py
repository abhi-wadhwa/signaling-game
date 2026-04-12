"""
General signaling game representation.

A signaling game consists of:
  - Nature: selects sender type theta from a prior distribution
  - Sender: observes theta, chooses a signal m from available signals
  - Receiver: observes m (but not theta), chooses an action a
  - Payoffs: u_S(theta, m, a) and u_R(theta, m, a)

A Perfect Bayesian Equilibrium (PBE) is:
  - Sender strategy: sigma(m | theta) for each type
  - Receiver strategy: alpha(a | m) for each signal
  - Belief system: mu(theta | m) for each signal
  satisfying sequential rationality and Bayesian consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SenderType:
    """A sender type in the signaling game."""

    name: str
    index: int

    def __hash__(self) -> int:
        return hash((self.name, self.index))


@dataclass(frozen=True)
class Signal:
    """A signal (message) the sender can choose."""

    name: str
    index: int

    def __hash__(self) -> int:
        return hash((self.name, self.index))


@dataclass(frozen=True)
class Action:
    """An action the receiver can choose."""

    name: str
    index: int

    def __hash__(self) -> int:
        return hash((self.name, self.index))


@dataclass
class PBE:
    """
    A Perfect Bayesian Equilibrium.

    Attributes:
        sender_strategy: dict mapping (type_index, signal_index) -> probability
        receiver_strategy: dict mapping (signal_index, action_index) -> probability
        beliefs: dict mapping (signal_index, type_index) -> probability
        sender_payoffs: dict mapping type_index -> expected payoff
        receiver_payoffs: dict mapping type_index -> expected payoff (conditional on type)
        equilibrium_type: 'separating', 'pooling', or 'semi-separating'
        label: optional human-readable description
    """

    sender_strategy: dict[tuple[int, int], float]
    receiver_strategy: dict[tuple[int, int], float]
    beliefs: dict[tuple[int, int], float]
    sender_payoffs: dict[int, float] = field(default_factory=dict)
    receiver_payoffs: dict[int, float] = field(default_factory=dict)
    equilibrium_type: str = "unknown"
    label: str = ""

    def get_sender_signal(self, type_idx: int) -> dict[int, float]:
        """Return the distribution over signals for a given type."""
        result = {}
        for (t, m), prob in self.sender_strategy.items():
            if t == type_idx and prob > 0:
                result[m] = prob
        return result

    def get_receiver_action(self, signal_idx: int) -> dict[int, float]:
        """Return the distribution over actions for a given signal."""
        result = {}
        for (m, a), prob in self.receiver_strategy.items():
            if m == signal_idx and prob > 0:
                result[a] = prob
        return result

    def get_belief(self, signal_idx: int) -> dict[int, float]:
        """Return the belief distribution over types given a signal."""
        result = {}
        for (m, t), prob in self.beliefs.items():
            if m == signal_idx:
                result[t] = prob
        return result

    def is_on_path(self, signal_idx: int) -> bool:
        """Check if a signal is on the equilibrium path."""
        total = 0.0
        for (t, m), prob in self.sender_strategy.items():
            if m == signal_idx:
                total += prob
        return total > 1e-12

    def classify(self, num_types: int, num_signals: int) -> str:
        """Classify the equilibrium as separating, pooling, or semi-separating."""
        # Gather each type's effective signal
        type_signals: dict[int, set[int]] = {}
        for t in range(num_types):
            sigs = set()
            for (tt, m), prob in self.sender_strategy.items():
                if tt == t and prob > 1e-12:
                    sigs.add(m)
            type_signals[t] = sigs

        # Separating: each type sends a distinct signal with probability 1
        all_pure = all(len(s) == 1 for s in type_signals.values())
        if all_pure:
            chosen = [next(iter(s)) for s in type_signals.values()]
            if len(set(chosen)) == len(chosen):
                self.equilibrium_type = "separating"
                return "separating"

        # Pooling: all types send the same signal
        if all_pure:
            chosen = [next(iter(s)) for s in type_signals.values()]
            if len(set(chosen)) == 1:
                self.equilibrium_type = "pooling"
                return "pooling"

        # Also check pooling when not pure but effectively same
        all_signals_same = True
        ref_dist = None
        for t in range(num_types):
            dist = {}
            for (tt, m), prob in self.sender_strategy.items():
                if tt == t:
                    dist[m] = prob
            if ref_dist is None:
                ref_dist = dist
            else:
                if set(dist.keys()) != set(ref_dist.keys()):
                    all_signals_same = False
                    break
                for m in dist:
                    if abs(dist[m] - ref_dist.get(m, 0)) > 1e-9:
                        all_signals_same = False
                        break

        if all_signals_same and ref_dist is not None:
            self.equilibrium_type = "pooling"
            return "pooling"

        self.equilibrium_type = "semi-separating"
        return "semi-separating"


@dataclass
class SignalingGame:
    """
    A finite signaling game.

    The payoff functions take (type_index, signal_index, action_index) -> float.
    """

    types: list[SenderType]
    signals: list[Signal]
    actions: list[Action]
    prior: np.ndarray  # probability distribution over types
    sender_payoff: Callable[[int, int, int], float] | np.ndarray
    receiver_payoff: Callable[[int, int, int], float] | np.ndarray

    def __post_init__(self) -> None:
        assert len(self.prior) == len(self.types)
        assert abs(sum(self.prior) - 1.0) < 1e-9
        assert all(p >= 0 for p in self.prior)

    @property
    def num_types(self) -> int:
        return len(self.types)

    @property
    def num_signals(self) -> int:
        return len(self.signals)

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def u_s(self, type_idx: int, signal_idx: int, action_idx: int) -> float:
        """Sender payoff."""
        if callable(self.sender_payoff):
            return self.sender_payoff(type_idx, signal_idx, action_idx)
        return float(self.sender_payoff[type_idx, signal_idx, action_idx])

    def u_r(self, type_idx: int, signal_idx: int, action_idx: int) -> float:
        """Receiver payoff."""
        if callable(self.receiver_payoff):
            return self.receiver_payoff(type_idx, signal_idx, action_idx)
        return float(self.receiver_payoff[type_idx, signal_idx, action_idx])

    def bayes_update(
        self,
        signal_idx: int,
        sender_strategy: dict[tuple[int, int], float],
    ) -> dict[int, float]:
        """
        Compute posterior beliefs mu(theta | m) via Bayes' rule.

        Returns a dict mapping type_index -> posterior probability.
        If the signal is off-path (zero probability), returns empty dict.
        """
        numerators = {}
        for t in range(self.num_types):
            sigma_m_given_t = sender_strategy.get((t, signal_idx), 0.0)
            numerators[t] = self.prior[t] * sigma_m_given_t

        total = sum(numerators.values())
        if total < 1e-15:
            return {}  # off-path

        return {t: num / total for t, num in numerators.items()}

    def expected_receiver_payoff(
        self,
        signal_idx: int,
        action_idx: int,
        beliefs: dict[int, float],
    ) -> float:
        """E[u_R(theta, m, a)] under beliefs mu(theta|m)."""
        return sum(
            beliefs.get(t, 0.0) * self.u_r(t, signal_idx, action_idx)
            for t in range(self.num_types)
        )
