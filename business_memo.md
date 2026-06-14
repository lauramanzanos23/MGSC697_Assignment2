# Business Memo: Inventory Replenishment Agent — Deployment Recommendation

**To:** VP Supply Chain / Head of Retail Operations  
**From:** Analytics & AI Team  
**Date:** June 2026  
**Re:** Q-Learning Agent for Weekly Stock Replenishment — Shadow Deployment Recommended  

---

## Recommendation

**Shadow the agent for one selling season before giving it authority over purchase orders.**

The agent has demonstrated strong performance in simulation and learned a policy that dominates both random ordering and the existing rule-based system on service level. However, the gap in variance, the fidelity limits of the simulator, and the absence of supplier constraints mean we should observe its decisions alongside — not instead of — the current system before committing.

---

## What we built and why it matters

We trained a tabular Q-learning agent on a discrete-state MDP that models weekly inventory replenishment. The agent observes two variables — inventory bin level and demand regime — and chooses how many units to order from a set of fixed quantities. It is trained over 1,000 simulated "years" (each 52 weeks) to maximize the net present value of weekly profit: sales revenue minus holding costs, stockout penalties, and ordering costs.

This is the right problem framing for a repeating, consequence-heavy decision that no rule can fully anticipate. The current (s,S) policy — order up to 30 units whenever stock falls below 10 — is a reasonable heuristic, but it was hand-tuned for medium demand and does not adapt when the demand regime shifts.

---

## What the evaluation showed

| Metric | Q-Learning | (s,S) Rule | Random |
|---|---|---|---|
| Mean annual return | +$8,924 | +$8,882 | +$7,223 |
| Std deviation | $2,120 | $1,612 | $1,555 |
| Fill rate | **99.76%** | 97.11% | 93.15% |
| Stockout rate | **0.62%** | 9.92% | 12.95% |
| Revenue diff vs (s,S): p-value | **p=0.32 (ns)** | — | — |
| High-demand start stress (50 eps) | **+$9,509** | +$9,372 | +$7,189 |

Three findings stand out.

First, the Q-learning agent nearly eliminates stockouts. A 0.62% stockout rate versus 9.9% for the rule-based system is not marginal — it means the agent serves roughly 14 more customers out of every 150 who would otherwise find an empty shelf. At $20 per unit sold, that is approximately $280 in recovered weekly revenue at a store doing 150 transactions per week. In a branded retail context, each stockout also carries a downstream cost (lost loyalty, substitute brand trial) that a $8/unit penalty in simulation only partially captures.

Second, a paired statistical test on 200 matched episodes confirms the revenue difference between Q-learning (+$8,924) and (s,S) (+$8,882) is not significant (p=0.32; 95% CI for the difference: −$42 to +$126). The agent does not sacrifice revenue to achieve its stockout advantage. It is timing orders more precisely around demand regime signals the rule ignores — not buying its way to a better fill rate by overstocking.

Third, when episodes start in a high-demand regime with depleted stock — the hardest opening condition — the agent pulls further ahead (+$9,509 vs +$9,372). The per-regime stockout analysis (fig7) confirms the advantage concentrates in the high-demand regime: the (s,S) rule was calibrated for medium demand and cannot adapt when conditions shift. The agent learned regime-specific order quantities that the static rule cannot replicate.

---

## What could go wrong in production

**Reward hacking.** The agent learned to minimize stockouts aggressively. In simulation this is correct. In production, if the supplier fills orders in fewer than the assumed units-of-five, or if warehouse capacity is actually tighter than the 50-unit ceiling in our model, the agent could trigger excessive order frequency and inflate costs in ways the reward function does not penalize. Constraint handling was implicit, not explicit.

**Higher variance.** The Q-learning return distribution has a standard deviation of $2,120 versus $1,612 for the (s,S) rule. In the worst decile (P10), Q-learning underperforms the rule by ~$608 per year (+$6,084 vs +$6,692). A buyer who is evaluated on worst-case outcomes — not expected value — has a rational reason to prefer the rule. This needs a management conversation before rollout.

**Simulator gap.** The environment models demand as a stationary Poisson process with a slow Markov regime switch. Real demand has calendar effects (weekends, holidays, promotions), correlated across SKUs, and subject to supply-side shocks. The agent was never trained on any of these. Its learned policy may be brittle to exactly the distribution shifts that matter most operationally.

**State observability.** The MDP assumes the demand regime is known. In practice, regime is latent — it has to be inferred from recent sales history. If regime classification lags by even two weeks, the agent's state representation is stale, and the Q-values it consults are for the wrong situation.

**Non-stationarity.** The Q-table is fixed after training. If mean demand drifts upward over several months (new store format, neighborhood growth), the learned policy does not update. A static table that was optimal last season can become suboptimal silently.

---

## Governance conditions for shadow deployment

Before we give the agent any authority over real orders, we need four things in place.

One: a real-time dashboard comparing the agent's recommended order to the buyer's actual order, with a flag whenever they diverge by more than one order tier. This surfaces distributional shift early and keeps the buyer informed.

Two: a hard cap rule: the agent may never recommend an order that would push inventory above physical shelf capacity (70 units for this SKU class). The simulator enforced a 50-unit ceiling; production needs the override at the warehouse level, not just in code.

Three: a regime-labeling pipeline. Before the agent can be trusted in production, we need a rolling demand classifier that tags each period's regime in real time. Until that is operational, the agent is making decisions with incomplete state.

Four: a six-month re-training cycle. Train on real transaction data once a season, evaluate on held-out weeks before deploying the updated table. Never update live.

---

## Bottom line

The agent has learned something real. Its stockout reduction is meaningful, its efficiency is comparable to a well-tuned rule, and its stress-test performance is encouraging. But simulation performance is not production performance. Shadow it for one season, instrument it properly, and then make the authority decision with real data rather than simulated confidence.

The question the course asks us to pose — *if this agent learns the wrong policy, who notices, and when?* — currently has the answer: no one, immediately. Shadow deployment is how we build the answer before we need it.
