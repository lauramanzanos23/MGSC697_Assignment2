"""
Evaluation and plotting
========================
Generates all figures required by the assignment rubric:
  1. Training reward curve (smoothed)
  2. Policy behaviour heatmap (Q-learning greedy orders)
  3. Evaluation comparison: distribution of episode returns
  4. Per-step component breakdown (one example episode)
  5. TD-error convergence
  6. Edge-case bar chart (high-demand start stress test)
  7. Per-regime stockout rate breakdown
  8. Sudden demand-shock stress test

Usage
-----
    python evaluate.py          # runs training first, then plots
    python evaluate.py --skip-training   # loads saved q_table.npy
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from environment import (
    InventoryEnv, ORDER_QUANTITIES, N_STATES,
    N_INV_BINS, N_DEMAND_REGIMES, INV_BINS, decode_state
)
from agents import QLearningAgent, SsAgent, RandomAgent
from train import train_and_evaluate, evaluate_agents, run_episode


# ─── Significance test (no scipy dependency) ─────────────────────────────────

def paired_ttest(a: list, b: list):
    """
    Paired t-test: H0: mean(a - b) == 0.
    Returns (t_stat, p_two_tailed, ci_low, ci_high) for 95% CI.
    Pure NumPy — no scipy required.
    """
    d    = np.array(a) - np.array(b)
    n    = len(d)
    mean = np.mean(d)
    se   = np.std(d, ddof=1) / np.sqrt(n)
    t    = mean / se
    # t critical for 95% two-tailed, df=n-1 (approximated for large n)
    # For n=200, df=199: t_crit ≈ 1.972  (using standard table value)
    t_crit = 1.972
    ci_low  = mean - t_crit * se
    ci_high = mean + t_crit * se
    # p-value approximation: 2 * P(T > |t|) using normal approx for large n
    import math
    def norm_sf(x):
        return 0.5 * (1 - math.erf(x / math.sqrt(2)))
    p = 2 * norm_sf(abs(t))
    return float(t), float(p), float(ci_low), float(ci_high)


FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

COLORS = {
    "q_learning": "#2563EB",   # blue
    "s_S_rule":   "#16A34A",   # green
    "random":     "#DC2626",   # red
}
LABELS = {
    "q_learning": "Q-Learning",
    "s_S_rule":   "(s,S) Rule-Based",
    "random":     "Random",
}


# ─── Helper ──────────────────────────────────────────────────────────────────

def smooth(values, window=50):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save(fig, name):
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ─── Figure 1: Training curve ────────────────────────────────────────────────

def plot_training_curve(train_returns: list, eval_mean: float):
    fig, ax = plt.subplots(figsize=(9, 4))

    raw  = np.array(train_returns)
    sm   = smooth(raw, window=50)
    x_sm = np.arange(len(sm)) + 50

    ax.plot(raw, color="#93C5FD", lw=0.6, alpha=0.6, label="Episode return")
    ax.plot(x_sm, sm, color=COLORS["q_learning"], lw=2.0, label="50-ep rolling mean")
    ax.axhline(eval_mean, color=COLORS["q_learning"], lw=1.5,
               ls="--", label=f"Eval mean (greedy) = {eval_mean:+.0f}")

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Total return ($)")
    ax.set_title("Q-Learning: Training Reward Curve")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig1_training_curve.png")


# ─── Figure 2: Learned policy heatmap ────────────────────────────────────────

def plot_policy_heatmap(q_agent: QLearningAgent):
    regime_names = ["Low demand", "Medium demand", "High demand"]
    inv_labels   = [f"{INV_BINS[i]}–{INV_BINS[i+1]-1}" for i in range(N_INV_BINS)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for reg in range(N_DEMAND_REGIMES):
        data = np.zeros(N_INV_BINS)
        for inv_bin in range(N_INV_BINS):
            from environment import state_index
            s = state_index(inv_bin, reg)
            data[inv_bin] = ORDER_QUANTITIES[np.argmax(q_agent.Q[s])]

        ax = axes[reg]
        bars = ax.barh(range(N_INV_BINS), data, color=COLORS["q_learning"],
                       alpha=0.75, edgecolor="white")
        ax.set_yticks(range(N_INV_BINS))
        ax.set_yticklabels(inv_labels, fontsize=8)
        ax.set_xlabel("Order quantity (units)")
        ax.set_title(regime_names[reg], fontweight="bold")
        ax.set_xlim(0, max(ORDER_QUANTITIES) + 3)
        ax.grid(True, axis="x", alpha=0.3)

        for bar, val in zip(bars, data):
            if val > 0:
                ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                        str(int(val)), va="center", fontsize=7)

    axes[0].set_ylabel("Inventory bin (units)")
    fig.suptitle("Learned Policy: Greedy Order Quantity by State", fontsize=12)
    fig.tight_layout()
    save(fig, "fig2_policy_heatmap.png")


# ─── Figure 3: Return distributions ─────────────────────────────────────────

def plot_return_distributions(eval_results: dict):
    fig, ax = plt.subplots(figsize=(9, 4))

    for name, res in eval_results.items():
        returns = res["returns"]
        ax.hist(
            returns,
            bins=30,
            alpha=0.55,
            color=COLORS[name],
            label=(
                f"{LABELS[name]}  "
                f"μ={res['mean']:+.0f}  "
                f"σ={res['std']:.0f}"
            ),
            density=True,
        )
        ax.axvline(res["mean"], color=COLORS[name], lw=1.5, ls="--")

    ax.set_xlabel("Episode total return ($)")
    ax.set_ylabel("Density")
    ax.set_title("Evaluation: Distribution of Episode Returns (200 episodes)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig3_return_distributions.png")


# ─── Figure 4: Per-step breakdown (one episode) ──────────────────────────────

def plot_episode_breakdown(q_agent: QLearningAgent):
    env = InventoryEnv(seed=99)
    result = run_episode(env, q_agent, train=False)
    infos  = result["infos"]

    steps     = [i["step"] for i in infos]
    inv_end   = [i["end_inventory"] for i in infos]
    demand    = [i["demand"] for i in infos]
    stockout  = [i["stockout"] for i in infos]
    orders    = [i["order_qty"] for i in infos]
    reward    = [i["revenue"] - i["holding"] - i["stockout_pen"] - i["order_cost"]
                 for i in infos]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(steps, inv_end, color=COLORS["q_learning"], lw=1.5, label="End inventory")
    axes[0].bar(steps, orders, alpha=0.4, color="#93C5FD", label="Order qty", width=0.8)
    axes[0].plot(steps, demand, color="#F59E0B", lw=1.2, ls="--", label="Demand")
    axes[0].set_ylabel("Units")
    axes[0].set_title("Example Episode — Q-Learning Policy")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(steps, stockout, color=COLORS["random"], alpha=0.7, label="Stockout (units)")
    axes[1].set_ylabel("Units short")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, reward, color=COLORS["s_S_rule"], lw=1.2)
    axes[2].axhline(0, color="black", lw=0.8, ls="--")
    axes[2].set_ylabel("Step reward ($)")
    axes[2].set_xlabel("Week")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "fig4_episode_breakdown.png")


# ─── Figure 5: TD-error convergence ─────────────────────────────────────────

def plot_td_convergence(q_agent: QLearningAgent):
    td = np.array(q_agent.td_errors)
    sm = smooth(td, window=500)
    x_sm = np.arange(len(sm)) + 500

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(td, color="#CBD5E1", lw=0.4, alpha=0.5, label="|TD error|")
    ax.plot(x_sm, sm, color=COLORS["q_learning"], lw=2.0, label="500-step rolling mean")

    ax.set_xlabel("Training step")
    ax.set_ylabel("|TD error|")
    ax.set_title("Q-Learning: TD Error Convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    save(fig, "fig5_td_convergence.png")


# ─── Figure 6: Edge-case bar chart ───────────────────────────────────────────

def plot_edge_case(edge_results: dict):
    names  = list(edge_results.keys())
    means  = [edge_results[n]["mean"] for n in names]
    stds   = [edge_results[n]["std"]  for n in names]
    colors = [COLORS[n] for n in names]
    labels = [LABELS[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=stds, color=colors,
                  capsize=6, alpha=0.8, edgecolor="white")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 20,
                f"{m:+.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean episode return ($)")
    ax.set_title("Edge Case: High-Demand Start Stress Test (50 episodes)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig6_edge_case.png")


# ─── Figure 7: Per-regime stockout breakdown ─────────────────────────────────

def plot_regime_breakdown(eval_results: dict):
    """
    Grouped bar chart: stockout rate per demand regime (Low/Med/High)
    for each agent.  Demonstrates that Q-learning's advantage is concentrated
    in the high-demand regime where the (s,S) rule struggles most.
    """
    regime_labels = ["Low demand\n(μ=5)", "Medium demand\n(μ=10)", "High demand\n(μ=18)"]
    agent_keys    = list(eval_results.keys())
    n_agents      = len(agent_keys)
    n_regimes     = 3
    x             = np.arange(n_regimes)
    width         = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, name in enumerate(agent_keys):
        regime_so = eval_results[name].get("regime_stockout", {})
        values    = [regime_so.get(r, 0) * 100 for r in range(n_regimes)]
        offset    = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=LABELS[name],
                      color=COLORS[name], alpha=0.82, edgecolor="white")
        for bar, v in zip(bars, values):
            if v > 0.3:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=10)
    ax.set_ylabel("Stockout rate (%)")
    ax.set_title("Stockout Rate by Demand Regime (200 paired episodes)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()
    save(fig, "fig7_regime_breakdown.png")


# ─── Figure 8: Sudden demand-shock stress test ───────────────────────────────

def plot_shock_test(shock_results: dict):
    """
    Bar chart comparing all three agents on the sudden demand-shock scenario:
    episodes start in low-demand (regime=0), then switch to high-demand at
    week 7.  Agents that use regime information in the state can adapt
    immediately; (s,S) cannot observe the regime change at all.
    """
    names  = list(shock_results.keys())
    means  = [shock_results[n]["mean"] for n in names]
    stds   = [shock_results[n]["std"]  for n in names]
    colors = [COLORS[n] for n in names]
    labels = [LABELS[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=stds, color=colors,
                  capsize=6, alpha=0.8, edgecolor="white")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 20,
                f"{m:+.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean episode return ($)")
    ax.set_title("Stress Test 2: Sudden Demand Shock\n"
                 "(Start regime=Low, forced to High at week 7 — 50 episodes)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig8_demand_shock.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main(skip_training: bool = False):
    """
    Two modes:
      Normal  : train the Q-learning agent, evaluate all three agents,
                produce all 8 figures.  Single training run — no duplication.
      Skip    : load saved q_table.npy and re-run evaluation with the same
                fixed seeds so numbers are reproducible.  Still generates all
                8 figures except fig1 (training curve) and fig5 (TD errors),
                which require a live training run and are skipped with a notice.
    """

    if skip_training and os.path.exists("q_table.npy"):
        # ── Skip-training path ────────────────────────────────────────────────
        # Load the saved Q-table and re-evaluate with identical fixed seeds.
        # Does NOT retrain, does NOT overwrite q_table.npy or results_summary.json.
        print("Loading saved Q-table …")
        q_agent   = QLearningAgent()
        q_agent.load("q_table.npy")
        ss_agent  = SsAgent()
        rnd_agent = RandomAgent(seed=42)
        # td_errors are not persisted to disk — fig5 will be skipped
        train_returns = None
        agents_map = {"q_learning": q_agent, "s_S_rule": ss_agent, "random": rnd_agent}
        eval_results, edge_results, shock_results = evaluate_agents(agents_map, verbose=True)

    else:
        # ── Full training path ────────────────────────────────────────────────
        results       = train_and_evaluate()
        q_agent       = results["q_agent"]        # live agent with td_errors
        train_returns = results["train_returns"]
        eval_results  = results["eval_results"]
        edge_results  = results["edge_results"]
        shock_results = results["shock_results"]

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures …")

    if train_returns is not None:
        plot_training_curve(train_returns, eval_results["q_learning"]["mean"])
    else:
        print("  (fig1 skipped in --skip-training mode; re-run without flag to regenerate)")

    plot_policy_heatmap(q_agent)
    plot_return_distributions(eval_results)
    plot_episode_breakdown(q_agent)

    if q_agent.td_errors:
        plot_td_convergence(q_agent)
    else:
        print("  (fig5 skipped — TD errors not available when loading saved Q-table)")

    plot_edge_case(edge_results)
    plot_regime_breakdown(eval_results)
    plot_shock_test(shock_results)

    print(f"\nAll figures saved to ./{FIGURE_DIR}/")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Evaluation summary ──────────────────────────────────────────────")
    print(f"{'Agent':15s} {'Mean':>8s} {'Std':>7s} {'P10':>8s} {'P90':>8s}"
          f"  {'Fill':>6s}  {'Stockout':>8s}")
    print("─" * 70)
    for name, res in eval_results.items():
        print(
            f"{LABELS[name]:15s} {res['mean']:+8.1f} {res['std']:7.1f} "
            f"{res['p10']:+8.1f} {res['p90']:+8.1f}  "
            f"{res['fill_rate']:6.2%}  {res['stockout_rate']:8.2%}"
        )

    # ── Significance tests ────────────────────────────────────────────────────
    print("\n── Paired t-tests (Q-learning vs baseline, n=200 matched episodes) ─")
    for baseline in ("s_S_rule", "random"):
        t, p, ci_lo, ci_hi = paired_ttest(
            eval_results["q_learning"]["returns"],
            eval_results[baseline]["returns"],
        )
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(
            f"  Q-learning vs {LABELS[baseline]:15s}: "
            f"Δmean={eval_results['q_learning']['mean']-eval_results[baseline]['mean']:+.1f}  "
            f"t={t:+.2f}  p={p:.4f} {sig}  "
            f"95% CI [{ci_lo:+.1f}, {ci_hi:+.1f}]"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    main(skip_training=args.skip_training)
