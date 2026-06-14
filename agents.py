"""
Agents for the Inventory Replenishment MDP
===========================================

Three agents, increasing sophistication:

1. RandomAgent         — uniformly random action selection (lower bound)
2. SsAgent             — rule-based (s, S) policy (strong heuristic baseline)
3. QLearningAgent      — tabular Q-learning with ε-greedy exploration

All agents share the same interface:
    agent.select_action(state) -> int   (action index)
    agent.update(...)           (no-op for non-learning agents)
"""

import numpy as np
from typing import Optional
from environment import (
    N_STATES, N_ACTIONS, ORDER_QUANTITIES,
    N_INV_BINS, N_DEMAND_REGIMES, INV_BINS,
    decode_state, DISCOUNT
)


# ─── Base ────────────────────────────────────────────────────────────────────

class BaseAgent:
    """Minimal interface every agent must implement."""

    def select_action(self, state: int) -> int:
        raise NotImplementedError

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """Return TD error (0.0 for non-learning agents)."""
        return 0.0

    def episode_reset(self):
        """Called at the start of each episode (optional)."""
        pass


# ─── 1. Random agent ─────────────────────────────────────────────────────────

class RandomAgent(BaseAgent):
    """
    Uniformly random policy.
    Serves as the floor baseline — any learning agent should beat this.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int) -> int:
        return int(self.rng.integers(0, N_ACTIONS))


# ─── 2. (s, S) rule-based agent ──────────────────────────────────────────────

class SsAgent(BaseAgent):
    """
    Classic (s, S) inventory policy:
        if inventory <= reorder_point  s:  order up-to level S
        else:                              do not order

    Parameters chosen to roughly match medium-demand regime:
        s = 10 (reorder when stock falls to ≤ 2 weeks of medium demand)
        S = 30 (order-up-to = ~3 weeks of medium demand)

    This is a strong, interpretable heuristic used in practice by most
    retail inventory systems without machine learning.
    """

    def __init__(self, s: int = 10, S: int = 30):
        self.s = s   # reorder point
        self.S = S   # order-up-to level

    def select_action(self, state: int) -> int:
        inv_bin, _ = decode_state(state)
        # Convert bin back to approximate midpoint inventory
        inv_mid = INV_BINS[inv_bin] + 2  # midpoint of bin

        if inv_mid <= self.s:
            desired_order = self.S - inv_mid
        else:
            desired_order = 0

        # Find closest available order quantity (round down to avoid overshoot)
        best_idx = 0
        for i, q in enumerate(ORDER_QUANTITIES):
            if q <= desired_order:
                best_idx = i
        return best_idx


# ─── 3. Tabular Q-learning agent ─────────────────────────────────────────────

class QLearningAgent(BaseAgent):
    """
    Tabular Q-learning (off-policy TD control).

    The agent maintains a Q-table Q[state, action] and updates it via:
        Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]

    This is the exact update derived in the course slides (slide 48).

    Exploration: ε-greedy with linear decay from ε_start → ε_min over
    the first `decay_episodes` training episodes.

    Parameters
    ----------
    alpha           : learning rate (step size)
    gamma           : discount factor
    epsilon_start   : initial exploration probability
    epsilon_min     : floor on exploration
    decay_episodes  : episodes over which ε decays linearly
    optimistic_init : initial Q-value (0 = pessimistic, >0 = optimistic)
    seed            : RNG seed
    """

    def __init__(
        self,
        alpha:           float = 0.10,
        gamma:           float = DISCOUNT,
        epsilon_start:   float = 1.0,
        epsilon_min:     float = 0.05,
        decay_episodes:  int   = 600,
        optimistic_init: float = 50.0,
        seed:            int   = 42,
    ):
        self.alpha          = alpha
        self.gamma          = gamma
        self.epsilon_start  = epsilon_start
        self.epsilon_min    = epsilon_min
        self.decay_episodes = decay_episodes
        self.rng            = np.random.default_rng(seed)

        # Optimistic initialisation encourages early exploration of all actions
        self.Q = np.full((N_STATES, N_ACTIONS), optimistic_init, dtype=float)

        self.episode_count = 0
        self._epsilon      = epsilon_start
        self.td_errors: list[float] = []

    # ── Epsilon schedule ─────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def episode_reset(self):
        self.episode_count += 1
        # Linear decay
        progress = min(self.episode_count / self.decay_episodes, 1.0)
        self._epsilon = self.epsilon_start + progress * (
            self.epsilon_min - self.epsilon_start
        )

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action(self, state: int) -> int:
        """ε-greedy: explore with probability ε, exploit otherwise."""
        if self.rng.random() < self._epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        return int(np.argmax(self.Q[state]))

    def greedy_action(self, state: int) -> int:
        """Pure greedy — used during evaluation."""
        return int(np.argmax(self.Q[state]))

    # ── Learning update ──────────────────────────────────────────────────────

    def update(
        self,
        state:      int,
        action:     int,
        reward:     float,
        next_state: int,
        done:       bool,
    ) -> float:
        """
        Q-learning (off-policy) update.
        Returns the absolute TD error for diagnostics.
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        td_error             = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        self.td_errors.append(abs(td_error))
        return abs(td_error)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def policy_summary(self) -> dict:
        """For each state, return the greedy action and max Q value."""
        summary = {}
        for s in range(N_STATES):
            best_a = int(np.argmax(self.Q[s]))
            summary[s] = {
                "greedy_action": ORDER_QUANTITIES[best_a],
                "max_Q":         float(self.Q[s, best_a]),
            }
        return summary

    def save(self, path: str):
        np.save(path, self.Q)

    def load(self, path: str):
        self.Q = np.load(path)
