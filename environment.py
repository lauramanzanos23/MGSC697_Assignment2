"""
Retail Inventory Replenishment MDP
===================================
A small, fully-specified Markov Decision Process for weekly store replenishment.

MDP summary
-----------
State  : (inventory_level_bin, demand_regime)
         inventory bins: [0,5), [5,10), ..., [40,45), [45,50] → 10 levels (0-9)
         demand regime : 0=low, 1=medium, 2=high              →  3 levels
         Total states  : 30

Action : order quantity index in {0, 1, 2, 3, 4, 5}
         maps to units  {0, 5, 10, 15, 20, 25}

Reward : sales_revenue
         - holding_cost   (0.50 $/unit/week for end-of-period inventory)
         - stockout_penalty (8.00 $/unit short — lost margin + goodwill)
         - ordering_cost  (fixed 10 + variable 2/unit, only when order > 0)

Transition:
         Demand is drawn from a Poisson distribution.
         Mean demand depends on the current demand regime.
         The regime itself evolves as a Markov chain (slow drift).

Horizon  : T = 52 steps (one year of weekly decisions)
Discount : γ = 0.97
"""

import numpy as np
from typing import Tuple, Dict, Any


# ─── Environment parameters ──────────────────────────────────────────────────

SELLING_PRICE   = 20.0   # $/unit
HOLDING_COST    = 0.50   # $/unit/week on ending inventory
STOCKOUT_PENALTY= 8.00   # $/unit of unmet demand
FIXED_ORDER_COST= 10.0   # $ flat cost per order placed
VARIABLE_ORDER_COST = 2.0  # $/unit ordered

MAX_INVENTORY   = 50     # physical capacity
HORIZON         = 52     # steps per episode

ORDER_QUANTITIES = [0, 5, 10, 15, 20, 25]  # discrete action set
N_ACTIONS        = len(ORDER_QUANTITIES)

# Demand regime: mean units demanded per week
DEMAND_MEANS    = {0: 5.0, 1: 10.0, 2: 18.0}   # low / medium / high
N_DEMAND_REGIMES = 3

# Regime transition matrix (slow drift)
REGIME_TRANSITION = np.array([
    [0.85, 0.13, 0.02],   # low   → low / medium / high
    [0.10, 0.80, 0.10],   # medium
    [0.02, 0.13, 0.85],   # high
])

# Inventory discretisation bins (right-exclusive except last)
# range(0,55,5) = [0,5,10,...,50] → 11 boundary values → 10 bins
INV_BINS = list(range(0, 55, 5))   # [0, 5, 10, ..., 50]
N_INV_BINS = len(INV_BINS) - 1     # 10 bins

DISCOUNT = 0.97


# ─── Helper ──────────────────────────────────────────────────────────────────

def inv_to_bin(inv: int) -> int:
    """Map continuous inventory level to discrete bin index."""
    for i in range(N_INV_BINS - 1):
        if inv < INV_BINS[i + 1]:
            return i
    return N_INV_BINS - 1  # overflow bin


def state_index(inv_bin: int, regime: int) -> int:
    return inv_bin * N_DEMAND_REGIMES + regime


def decode_state(idx: int) -> Tuple[int, int]:
    inv_bin = idx // N_DEMAND_REGIMES
    regime  = idx %  N_DEMAND_REGIMES
    return inv_bin, regime


N_STATES = N_INV_BINS * N_DEMAND_REGIMES   # 30  (10 bins × 3 regimes)


# ─── Environment class ───────────────────────────────────────────────────────

class InventoryEnv:
    """
    Discrete-time inventory replenishment environment.

    A "period" represents one week.  At the start of each period the agent
    places an order (arrives immediately — lead-time = 0 for simplicity),
    then demand is realised and the reward is computed.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self) -> int:
        """Reset to a random starting state.  Returns state index."""
        self.inventory   = int(self.rng.integers(5, 25))   # start somewhere mid-range
        self.regime      = 1                                # start in medium demand
        self.step_count  = 0
        self.done        = False
        self._history: list[Dict[str, Any]] = []
        return self._state()

    def step(self, action_idx: int) -> Tuple[int, float, bool, dict]:
        """
        Apply action and advance one period.

        Returns
        -------
        next_state  : int     state index after transition
        reward      : float   period reward (can be negative)
        done        : bool    True when horizon reached
        info        : dict    diagnostic data
        """
        assert not self.done, "Episode is over — call reset()"
        assert 0 <= action_idx < N_ACTIONS, f"Invalid action {action_idx}"

        order_qty = ORDER_QUANTITIES[action_idx]

        # 1. Receive order (lead-time = 0)
        self.inventory = min(self.inventory + order_qty, MAX_INVENTORY)

        # 2. Draw demand
        mean_demand = DEMAND_MEANS[self.regime]
        demand      = int(self.rng.poisson(mean_demand))

        # 3. Fulfil demand
        units_sold  = min(self.inventory, demand)
        stockout    = max(demand - self.inventory, 0)
        self.inventory -= units_sold

        # 4. Compute reward
        revenue      = units_sold * SELLING_PRICE
        holding      = self.inventory * HOLDING_COST
        stockout_pen = stockout * STOCKOUT_PENALTY
        if order_qty > 0:
            order_cost = FIXED_ORDER_COST + order_qty * VARIABLE_ORDER_COST
        else:
            order_cost = 0.0

        reward = revenue - holding - stockout_pen - order_cost

        # 5. Regime transition
        self.regime = int(self.rng.choice(
            N_DEMAND_REGIMES,
            p=REGIME_TRANSITION[self.regime]
        ))

        # 6. Advance time
        self.step_count += 1
        self.done = self.step_count >= HORIZON

        next_state = self._state()

        info = {
            "order_qty":    order_qty,
            "demand":       demand,
            "units_sold":   units_sold,
            "stockout":     stockout,
            "end_inventory":self.inventory,
            "regime":       self.regime,
            "revenue":      revenue,
            "holding":      holding,
            "stockout_pen": stockout_pen,
            "order_cost":   order_cost,
            "step":         self.step_count,
        }
        self._history.append(info)
        return next_state, reward, self.done, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _state(self) -> int:
        return state_index(inv_to_bin(self.inventory), self.regime)

    def action_space_size(self) -> int:
        return N_ACTIONS

    def state_space_size(self) -> int:
        return N_STATES

    def render_step(self, info: dict) -> str:
        """Human-readable one-liner for a step."""
        return (
            f"step={info['step']:3d} | "
            f"inv={info['end_inventory']:3d} | "
            f"order={info['order_qty']:3d} | "
            f"demand={info['demand']:3d} | "
            f"stockout={info['stockout']:3d} | "
            f"regime={'LMH'[info['regime']]} | "
            f"reward={info['revenue']-info['holding']-info['stockout_pen']-info['order_cost']:+7.1f}"
        )
