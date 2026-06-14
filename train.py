"""
Training loop
=============
Trains the Q-learning agent and collects episode returns for all three agents.

Usage
-----
    python train.py            # runs with defaults, saves Q-table + results
    python train.py --seed 7   # reproducible with different seed
"""

import argparse
import json
import time
import numpy as np

from environment import InventoryEnv
from agents import RandomAgent, SsAgent, QLearningAgent


# ─── Default hyperparameters ─────────────────────────────────────────────────

TRAIN_EPISODES  = 1000   # Q-learning training episodes
EVAL_EPISODES   = 200    # evaluation episodes (greedy policy, no exploration)
SEED_ENV        = 0
SEED_AGENT      = 42


# ─── Single episode runner ───────────────────────────────────────────────────

def run_episode(
    env: InventoryEnv,
    agent,
    train: bool = False,
    render_last: bool = False,
) -> dict:
    """
    Run one episode.

    Parameters
    ----------
    train       : if True, call agent.update() after every step
    render_last : print the last 5 steps to stdout

    Returns
    -------
    dict with total_reward, step_rewards, and per-step info list
    """
    agent.episode_reset()
    state      = env.reset()
    total_r    = 0.0
    step_rews  = []
    infos      = []

    while True:
        if train:
            action = agent.select_action(state)
        else:
            # evaluation: pure greedy if available, else default select_action
            action = (
                agent.greedy_action(state)
                if hasattr(agent, "greedy_action")
                else agent.select_action(state)
            )

        next_state, reward, done, info = env.step(action)
        step_rews.append(reward)
        infos.append(info)

        if train:
            agent.update(state, action, reward, next_state, done)

        total_r += reward
        state    = next_state

        if done:
            break

    if render_last:
        for info in infos[-5:]:
            print("  ", env.render_step(info))

    return {"total_reward": total_r, "step_rewards": step_rews, "infos": infos}


# ─── Main training routine ───────────────────────────────────────────────────

def train_and_evaluate(
    train_episodes: int = TRAIN_EPISODES,
    eval_episodes:  int = EVAL_EPISODES,
    seed_env:       int = SEED_ENV,
    seed_agent:     int = SEED_AGENT,
    verbose:        bool = True,
) -> dict:

    env       = InventoryEnv(seed=seed_env)
    q_agent   = QLearningAgent(seed=seed_agent)
    ss_agent  = SsAgent()
    rnd_agent = RandomAgent(seed=seed_agent)

    # ── Train Q-learning ─────────────────────────────────────────────────────
    if verbose:
        print(f"Training Q-learning agent for {train_episodes} episodes …")

    t0 = time.time()
    train_returns = []

    for ep in range(train_episodes):
        result = run_episode(env, q_agent, train=True)
        train_returns.append(result["total_reward"])

        if verbose and (ep + 1) % 200 == 0:
            recent = np.mean(train_returns[-200:])
            print(
                f"  ep {ep+1:5d}/{train_episodes} | "
                f"ε={q_agent.epsilon:.3f} | "
                f"mean return (last 200) = {recent:+.1f}"
            )

    elapsed = time.time() - t0
    if verbose:
        print(f"Training done in {elapsed:.1f}s\n")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    agents_map = {"q_learning": q_agent, "s_S_rule": ss_agent, "random": rnd_agent}
    eval_results, edge_results, shock_results = evaluate_agents(
        agents_map, eval_episodes=eval_episodes, verbose=verbose
    )

    # ── Save Q-table + JSON summary ──────────────────────────────────────────
    q_agent.save("q_table.npy")
    if verbose:
        print("\nQ-table saved to q_table.npy")

    summary = {k: {kk: vv for kk, vv in v.items() if kk != "returns"}
               for k, v in eval_results.items()}
    with open("results_summary.json", "w") as f:
        json.dump({"eval": summary, "edge": edge_results,
                   "shock": shock_results,
                   "training_time_s": elapsed}, f, indent=2)

    return {
        "train_returns":  train_returns,
        "eval_results":   eval_results,
        "edge_results":   edge_results,
        "shock_results":  shock_results,
        "training_time_s": elapsed,
        "q_agent":        q_agent,
        "hyperparams": {
            "train_episodes":  train_episodes,
            "eval_episodes":   eval_episodes,
            "alpha":           q_agent.alpha,
            "gamma":           q_agent.gamma,
            "epsilon_start":   q_agent.epsilon_start,
            "epsilon_min":     q_agent.epsilon_min,
            "decay_episodes":  q_agent.decay_episodes,
        }
    }


# ─── Standalone evaluation (used by evaluate.py --skip-training) ─────────────

def evaluate_agents(
    agents: dict,
    eval_episodes: int = EVAL_EPISODES,
    verbose: bool = True,
) -> tuple:
    """
    Evaluate a dict of {name: agent} and return (eval_results, edge_results).
    Uses fixed seeds throughout so results are reproducible regardless of
    whether training was run first.

    Parameters
    ----------
    agents        : dict mapping name → agent instance
    eval_episodes : number of normal evaluation episodes per agent
    verbose       : print progress

    Returns
    -------
    eval_results  : dict with per-agent mean, std, p10, p90, fill/stockout rates
    edge_results  : dict with per-agent mean/std for the high-demand start test
    shock_results : dict with per-agent mean/std for the sudden demand-shock test
    """
    if verbose:
        print(f"Evaluating all agents for {eval_episodes} episodes …")

    # Each episode i uses InventoryEnv(seed=1000+i) for ALL agents.
    # This ensures a paired comparison: every agent faces the exact same
    # demand realisations and starting inventories, so differences in
    # returns are attributable to policy, not to sampling variation.
    episode_seeds = [1000 + i for i in range(eval_episodes)]
    eval_results  = {}

    for name, agent in agents.items():
        returns    = []
        all_infos  = []
        regime_info = {0: [], 1: [], 2: []}  # per-regime step-level info

        for seed in episode_seeds:
            env_ep = InventoryEnv(seed=seed)
            r = run_episode(env_ep, agent, train=False)
            returns.append(r["total_reward"])
            all_infos.extend(r["infos"])
            for info in r["infos"]:
                regime_info[info["regime"]].append(info)

        stockout_rate = np.mean([i["stockout"] > 0 for i in all_infos])
        fill_rate     = np.mean([
            i["units_sold"] / max(i["units_sold"] + i["stockout"], 1)
            for i in all_infos
        ])

        # Per-regime stockout rate
        regime_stockout = {}
        for reg, infos in regime_info.items():
            if infos:
                regime_stockout[reg] = float(np.mean([i["stockout"] > 0 for i in infos]))
            else:
                regime_stockout[reg] = 0.0

        eval_results[name] = {
            "mean":            float(np.mean(returns)),
            "std":             float(np.std(returns)),
            "p10":             float(np.percentile(returns, 10)),
            "p90":             float(np.percentile(returns, 90)),
            "stockout_rate":   float(stockout_rate),
            "fill_rate":       float(fill_rate),
            "returns":         returns,
            "regime_stockout": regime_stockout,
        }
        if verbose:
            print(
                f"  {name:12s} | mean={eval_results[name]['mean']:+8.1f} "
                f"| std={eval_results[name]['std']:6.1f} "
                f"| fill_rate={fill_rate:.2%} "
                f"| stockout_rate={stockout_rate:.2%}"
            )

    # ── Edge-case: high-demand start stress test ──────────────────────────────
    # Each episode starts in high-demand regime with low inventory (5 units).
    # The regime can transition away after each step — this tests the agent's
    # ability to react quickly to demand spikes, not sustained high demand.
    # Seeds are fixed per episode (2000+i) for full reproducibility.
    if verbose:
        print("\nEdge-case: high-demand start stress test (50 episodes) …")

    edge_results = {}
    for name, agent in agents.items():
        returns = []
        for i in range(50):
            env2 = InventoryEnv(seed=2000 + i)
            env2.regime     = 2   # force high-demand start
            env2.inventory  = 5   # start with low stock
            env2.step_count = 0
            env2.done       = False

            total_r = 0.0
            state   = env2._state()
            agent.episode_reset()
            while True:
                action = (
                    agent.greedy_action(state)
                    if hasattr(agent, "greedy_action")
                    else agent.select_action(state)
                )
                next_state, reward, done, _ = env2.step(action)
                total_r += reward
                state    = next_state
                if done:
                    break
            returns.append(total_r)

        edge_results[name] = {
            "mean": float(np.mean(returns)),
            "std":  float(np.std(returns)),
        }
        if verbose:
            print(
                f"  {name:12s} | edge mean={edge_results[name]['mean']:+8.1f} "
                f"| std={edge_results[name]['std']:6.1f}"
            )

    # ── Stress-test 2: sudden demand shock ────────────────────────────────────
    # Each episode starts in low-demand regime (regime=0) with normal stock
    # (20 units). At step 6, the regime is forced to high-demand (regime=2)
    # — simulating a sudden promotional event or competitor exit. This tests
    # whether the agent's state representation allows it to adapt quickly when
    # it observes the regime change in the state vector.
    # Seeds are fixed per episode (3000+i) for full reproducibility.
    if verbose:
        print("\nStress-test 2: sudden demand shock (50 episodes) …")

    shock_results = {}
    for name, agent in agents.items():
        returns = []
        for i in range(50):
            env3 = InventoryEnv(seed=3000 + i)
            env3.regime     = 0    # start low demand
            env3.inventory  = 20   # normal starting stock
            env3.step_count = 0
            env3.done       = False

            total_r = 0.0
            state   = env3._state()
            agent.episode_reset()
            step    = 0
            while True:
                # At step 6 (week 7), force the regime to high demand
                if step == 6:
                    env3.regime = 2
                    state = env3._state()   # state now reflects new regime

                action = (
                    agent.greedy_action(state)
                    if hasattr(agent, "greedy_action")
                    else agent.select_action(state)
                )
                next_state, reward, done, _ = env3.step(action)
                total_r += reward
                state    = next_state
                step    += 1
                if done:
                    break
            returns.append(total_r)

        shock_results[name] = {
            "mean": float(np.mean(returns)),
            "std":  float(np.std(returns)),
        }
        if verbose:
            print(
                f"  {name:12s} | shock mean={shock_results[name]['mean']:+8.1f} "
                f"| std={shock_results[name]['std']:6.1f}"
            )

    return eval_results, edge_results, shock_results


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--eval-episodes",  type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed",           type=int, default=SEED_ENV)
    args = parser.parse_args()

    train_and_evaluate(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seed_env=args.seed,
    )
