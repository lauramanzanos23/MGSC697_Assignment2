# Assignment 2 — Sequential Decision Agent
## Retail Inventory Replenishment via Tabular Q-Learning

**MGSC 697 · Designing & Building Agentic AI Systems**

---

## What this project does

Trains and evaluates a tabular Q-learning agent on a weekly retail inventory replenishment MDP. The agent decides how many units to order each week to maximise annual profit — balancing stockout penalties, holding costs, and ordering costs.

---

## Project structure

```
.
├── environment.py      # MDP definition: state, action, reward, transitions
├── agents.py           # RandomAgent, SsAgent (rule-based), QLearningAgent
├── train.py            # Training loop + evaluation runner
├── evaluate.py         # All eight figures + summary table + significance tests
├── business_memo.md    # 1.5-page deploy/shadow/reject recommendation
├── requirements.txt    # numpy, matplotlib (pinned)
├── q_table.npy         # Saved Q-table (skip re-training)
├── results_summary.json
└── figures/
    ├── fig1_training_curve.png
    ├── fig2_policy_heatmap.png
    ├── fig3_return_distributions.png
    ├── fig4_episode_breakdown.png
    ├── fig5_td_convergence.png
    ├── fig6_edge_case.png
    ├── fig7_regime_breakdown.png
    └── fig8_demand_shock.png
```

---

## Setup

**Requirements:** Python 3.11+, NumPy ≥ 1.24, Matplotlib ≥ 3.7 (no RL framework needed).

```bash
pip install -r requirements.txt
```

---

## How to run

**Full run — train and plot (≈ 2 seconds on any laptop):**
```bash
python evaluate.py
```

**Training only:**
```bash
python train.py
# optional flags:
python train.py --train-episodes 2000 --eval-episodes 500 --seed 7
```

**Skip training, use saved Q-table:**
```bash
python evaluate.py --skip-training
```

All figures are saved to `./figures/`. A results summary is written to `results_summary.json`.

---

## MDP specification

| Component | Design |
|---|---|
| **State** | `(inventory_bin, demand_regime)` — 10 bins × 3 regimes = **30 states** |
| **Action** | Order quantity ∈ {0, 5, 10, 15, 20, 25} — **6 discrete actions** |
| **Reward** | Sales revenue − holding cost − stockout penalty − ordering cost |
| **Transition** | Demand ~ Poisson(μ_regime); regime evolves via Markov chain |
| **Horizon** | T = 52 steps (one year of weekly decisions) |
| **Discount γ** | 0.97 |

**Reward components:**
- Selling price: $20/unit sold
- Holding cost: $0.50/unit remaining at end of period
- Stockout penalty: $8.00/unit of unmet demand
- Ordering cost: $10 fixed + $2/unit (only when order > 0)

**State space design:** Inventory is discretised into 10 bins of width 5 (0–4, 5–9, …, 45–50). The demand regime (Low/Medium/High) is treated as observable in training; in production it would need to be inferred from sales history (see Failure Analysis). Starting inventory is drawn uniformly from [5, 25) at the beginning of each training episode; demand regime starts at Medium.

**Demand regimes:**

| Regime | Mean weekly demand | Transition |
|---|---|---|
| Low | 5 units | Self-loop 85% |
| Medium | 10 units | Self-loop 80% |
| High | 18 units | Self-loop 85% |

**Why Poisson demand?** Poisson is the canonical model for customer arrivals at a retail shelf: inter-arrivals are memoryless, variance equals mean, and it has a single interpretable parameter (mean weekly footfall × conversion rate). The Markov chain over regimes captures slow seasonal drift without calendar complexity.

---

## Agents

### Baseline 1 — Random
Uniformly samples an order quantity at each step. Lower bound on performance.

### Baseline 2 — (s, S) Rule-Based
Industry-standard heuristic: order up to S=30 units whenever inventory ≤ s=10. Parameters chosen to match medium demand (s=10 ≈ two weeks of μ=5, S=30 ≈ three weeks of μ=10). Strong and interpretable — this is what most real retail inventory systems run without machine learning.

### Learning agent — Tabular Q-Learning

Updates Q(s,a) via the off-policy TD (Bellman) rule:

```
Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

**Why Q-learning (off-policy) rather than SARSA (on-policy)?** Q-learning learns the value of the *greedy* policy regardless of which action was actually taken during exploration. This is advantageous here because the evaluation policy (pure greedy) differs from the training policy (ε-greedy). SARSA would learn the value of the mixed exploration policy, which is not what we want to deploy. The off-policy target `max_a' Q(s',a')` also tends to converge faster in tabular settings by always bootstrapping from the best known action.

**Convergence:** Tabular Q-learning converges to Q* under the Robbins–Monro conditions (∑α=∞, ∑α²<∞) provided all state-action pairs are visited infinitely often. We use a constant α=0.10 (which does not satisfy Robbins–Monro strictly), accepted in practice because: (1) the state space is small (30 states × 6 actions = 180 cells), (2) optimistic initialisation (+50) plus full exploration (ε=1.0 → 0.05) ensures all cells are visited many times, and (3) the TD-error convergence plot (fig5) confirms empirical stability.

Key hyperparameters:

| Parameter | Value | Rationale |
|---|---|---|
| α (learning rate) | 0.10 | Standard for tabular setting |
| γ (discount) | 0.97 | Weights ~1 year of future reward |
| ε start | 1.0 | Full exploration at episode 1 |
| ε min | 0.05 | Residual exploration for non-stationarity |
| ε decay | linear over 600 ep | Balances exploration vs convergence |
| Optimistic init | +50 | Encourages visiting all state-action pairs |
| Training episodes | 1,000 | Convergence confirmed via TD-error plot |

---

## Evaluation methodology

All three agents are evaluated on the **same 200 episodes**: episode *i* uses `InventoryEnv(seed=1000+i)` for every agent. This paired design eliminates sampling variation as a confounder and enables paired significance testing. Results are not affected by training-time RNG state.

---

## Evaluation results

### Normal operation (200 paired evaluation episodes, greedy policy)

| Agent | Mean return | Std | P10 | P90 | Fill rate | Stockout rate |
|---|---|---|---|---|---|---|
| **Q-Learning** | **+$8,924** | $2,120 | +$6,084 | +$12,027 | **99.76%** | **0.62%** |
| (s,S) Rule | +$8,882 | $1,612 | +$6,692 | +$11,371 | 97.11% | 9.92% |
| Random | +$7,223 | $1,555 | +$5,319 | +$9,191 | 93.15% | 12.95% |

### Statistical significance (paired t-tests, n=200)

| Comparison | Δ mean | t-stat | p-value | 95% CI |
|---|---|---|---|---|
| Q-Learning vs (s,S) Rule | +$42 | 0.99 | 0.32 (ns) | [−$42, +$126] |
| Q-Learning vs Random | +$1,701 | 18.34 | <0.001 *** | [+$1,518, +$1,884] |

**The revenue difference between Q-learning and (s,S) is statistically insignificant** (p=0.32). This is the correct result: the agents are comparable on revenue. The material advantage is in service level — a 16× reduction in stockout rate (0.62% vs 9.92%) that does not show up in the return comparison because the $8 penalty per missed unit is absorbed across a 52-week horizon.

### Stress tests

| Stress scenario | Q-Learning | (s,S) Rule | Random | Interpretation |
|---|---:|---:|---:|---|
| High-demand start (inv=5, regime=High at t=0) | **+$9,509** | +$9,372 | +$7,189 | Agent adapts to demand spike faster than the rule |
| Sudden demand shock (regime Low→High at week 7) | **+$9,093** | +$8,945 | +$7,456 | Agent reads regime in state; (s,S) is regime-blind |

**High-demand start (50 episodes):** Each episode starts in the high-demand regime with only 5 units in stock. The regime can transition away after each step. Tests immediate response to a demand spike at the worst possible inventory level.

**Sudden demand shock (50 episodes):** Each episode starts in low demand (regime=0) with normal stock (20 units). At week 7, the regime is forced to high demand — simulating a promotional event or competitor exit. Because Q-learning’s state includes the demand regime, it observes the change immediately and adjusts orders. The (s,S) rule has no such signal and reacts only through inventory depletion, which takes 2–3 weeks to register.

**Key finding:** Q-learning achieves a 0.62% stockout rate vs 9.92% for the rule-based policy — a **16× reduction** — while being statistically indistinguishable on expected annual revenue (p=0.32). Both stress tests confirm the agent uses regime information productively; the (s,S) rule cannot adapt to conditions it was not calibrated for.

---

## Failure analysis

**1. Reward hacking — over-ordering risk.** The agent minimises stockout aggressively. If supplier minimum order units are smaller than the 5-unit discretisation assumed here, the agent could issue more frequent small orders and inflate fixed ordering costs ($10 flat per order). The reward function does not penalise order frequency directly.

**2. Higher variance.** The Q-learning return distribution has σ=$2,120 vs $1,612 for (s,S). The revenue difference is not significant (p=0.32), but in the worst decile (P10), Q-learning underperforms the rule by ~$608/year (+$6,084 vs +$6,692). Managers evaluated on downside risk may rationally prefer the rule even knowing the aggregate stockout advantage.

**3. Simulator gap.** The environment uses stationary Poisson demand with slow regime shifts. Real retail demand has calendar effects, cross-SKU correlation, and supply-side shocks. The agent never trained on these patterns and may be brittle to exactly the distribution shifts that matter most operationally.

**4. Latent state.** The demand regime is assumed observable in simulation. In production it is latent and must be inferred from recent sales history — a classification problem the current system does not address.

**5. Non-stationarity.** The Q-table is static after training. Sustained demand drift (seasonal growth, new competitors) will degrade performance silently and without warning.

---

## Figures

| Figure | What it shows |
|---|---|
| `fig1_training_curve.png` | Episode returns during training with 50-episode rolling mean |
| `fig2_policy_heatmap.png` | Greedy order quantity for every (inventory, regime) state |
| `fig3_return_distributions.png` | Return distributions for all three agents (200 paired episodes) |
| `fig4_episode_breakdown.png` | Per-week inventory, demand, stockouts, and reward for one episode |
| `fig5_td_convergence.png` | Absolute TD error over training steps (log scale) |
| `fig6_edge_case.png` | Mean returns ± std under high-demand start stress test |
| `fig7_regime_breakdown.png` | Stockout rate by demand regime — shows where each agent struggles |
| `fig8_demand_shock.png` | Sudden demand-shock stress test: regime forced Low→High at week 7 |

---

## Reproducibility

All runs use fixed seeds. Evaluation uses per-episode seeds (`InventoryEnv(seed=1000+i)` for episode `i`) so all three agents face identical demand trajectories. The saved `q_table.npy` reproduces the reported evaluation exactly:

```bash
python evaluate.py --skip-training
```

Training time: **≈ 1 second** (1,000 episodes × 52 steps on a MacBook).

---

## Rubric coverage

| Assignment requirement | Where addressed |
|---|---|
| Problem framing (state, action, reward, transition, horizon) | MDP specification section above; `environment.py` |
| Baseline agents | `RandomAgent` and `SsAgent` in `agents.py` |
| Learning agent | `QLearningAgent` in `agents.py`; Bellman update in train.py |
| Evaluation against baselines | `evaluate_agents()` in `train.py`; `evaluate.py`; `results_summary.json` |
| Edge / stress episodes | High-demand start (50 eps) + Sudden demand shock (50 eps) in `train.py` |
| Plots | `figures/` — 8 PNG files covering training curve, policy, distributions, convergence, stress tests |
| Failure analysis | Failure analysis section above; `business_memo.md` |
| Business memo (deploy/shadow/reject) | `business_memo.md` |

---

## Design decisions and course connection

The MDP design follows the **MDP design checklist** from the course slides (slide 32): decision point, transition, reward, horizon, constraints, and evaluator are all explicitly specified. The Q-learning update is the exact off-policy Bellman formula from slide 48.

The (s,S) baseline was chosen deliberately — it is not a trivially weak benchmark. Most real retail inventory systems use exactly this rule. Showing that Q-learning is statistically indistinguishable on revenue while cutting stockouts 16× is the meaningful result, not merely beating random.

The discount factor γ=0.97 encodes a medium-term time preference: approximately 33 future periods (weeks) of reward are weighted meaningfully. This aligns with a quarterly inventory planning horizon — the "high discount factor" column in the course's business interpretation table (slide 29).

Lead time is set to zero (orders arrive immediately) as a simplifying assumption explicitly acknowledged in the code. Relaxing this would require augmenting the state space with the pipeline (units on order) — a natural extension noted in the failure analysis under simulator gap.
