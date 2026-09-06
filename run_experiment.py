# -*- coding: utf-8 -*-
"""一键运行完整对比实验。

用法：
  python run_experiment.py --quick   # 快速冒烟（小规模，验证流程）
  python run_experiment.py           # 完整实验（默认参数）

实验设计：
  · N_SCENARIO 个场景（公共随机数 CRN）：同一场景下四种算法使用
    完全相同的 EV 出行链、意愿矩阵与基础负荷，保证公平对比
  · PPO 在随机场景流上训练（域随机化），在 CRN 场景上确定性评估
  · 输出：results/ 下的指标表、曲线数据、全部图表
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v2g.config import Config
from v2g.scenario import Scenario
from v2g.grid import Grid33
from v2g.willingness import willingness
from v2g.environment import V2GSim, V2GEnv
from v2g.metrics import evaluate_plan
from algorithms.uncoordinated import uncoordinated_plan
from algorithms.tou_response import tou_plan
from algorithms.mopso import MOPSO

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

METHODS = ["Uncoordinated", "TOU Response", "MOPSO", "PPO (ours)"]
COLORS = {"Uncoordinated": "#d62728", "TOU Response": "#ff7f0e",
          "MOPSO": "#1f77b4", "PPO (ours)": "#2ca02c"}


# ----------------------------------------------------------------------
def build_scenario(cfg, seed):
    """生成场景 + 参考轨迹意愿矩阵（所有算法共用）。"""
    rng = np.random.default_rng(seed)
    sc = Scenario(cfg, rng)
    sim0 = V2GSim(cfg, sc, np.zeros((sc.n_ev, cfg.T)))
    res0 = sim0.run(uncoordinated_plan(cfg, sim0))
    w = willingness(res0["soc"], sc.need_at_dep, sc.e_cap,
                    sc.t_dep_next, cfg.price_array(), cfg)
    return sc, w


def ppo_env_fn(cfg):
    """PPO 训练环境工厂：每个回合随机新场景（域随机化）。"""
    def _make():
        seed = int(np.random.default_rng().integers(0, 1e9))
        sc, w = build_scenario(cfg, seed)
        grid = Grid33(cfg)
        return V2GEnv(cfg, sc, w, grid)
    return _make


# ----------------------------------------------------------------------
def run(cfg, n_scen, quick=False):
    t_start = time.time()
    if quick:
        cfg.PSO_ITER, cfg.PSO_POP = 12, 12
        cfg.PPO_STEPS_TOTAL = 8_000
        cfg.PPO_BATCH = 4_000
        n_scen = min(n_scen, 2)

    all_metrics = {m: [] for m in METHODS}
    curves0 = {}          # 场景 0 的曲线（绘图用）
    pareto = None
    train_hist = None
    w0 = None             # 场景 0 意愿矩阵（绘图用）

    # ---------- PPO 训练（一次性，随机场景流）----------
    from algorithms import ppo as ppo_mod
    if ppo_mod.HAS_TORCH:
        print("[PPO] training on randomized scenarios ...")
        t0 = time.time()
        agent, ep_r, ep_v = ppo_mod.train(
            ppo_env_fn(cfg), cfg,
            log_fn=lambda s: print("  " + s))
        train_hist = {"reward": ep_r, "viol": ep_v,
                      "time_s": time.time() - t0}
        print(f"[PPO] trained in {train_hist['time_s']:.0f}s, "
              f"last-20ep reward {np.mean(ep_r[-20:]):.1f}, "
              f"viol {np.mean(ep_v[-20:]):.2f}")
        # 保存模型检查点（供大屏/复现使用）
        import torch as _torch
        os.makedirs(os.path.join(os.path.dirname(RESULTS), "models"),
                    exist_ok=True)
        _torch.save(agent.net.state_dict(),
                    os.path.join(os.path.dirname(RESULTS),
                                 "models", "ppo_agent.pt"))
    else:
        print("[WARN] torch 不可用，PPO 跳过")
        METHODS.remove("PPO (ours)")

    # ---------- 场景循环 ----------
    for si in range(n_scen):
        print(f"=== Scenario {si+1}/{n_scen} ===")
        sc, w = build_scenario(cfg, cfg.SEED + 1000 + si)
        if si == 0:
            w0 = w
        grid = Grid33(cfg)
        sim = V2GSim(cfg, sc, w)

        plans = {
            "Uncoordinated": uncoordinated_plan(cfg, sim),
            "TOU Response": tou_plan(cfg, sim, sc),
        }

        t0 = time.time()
        mopso = MOPSO(cfg, sc, w, grid, seed=si, verbose=(si == 0))
        r = mopso.solve()
        plans["MOPSO"] = r["plan"]
        print(f"  [MOPSO] done in {time.time()-t0:.0f}s, "
              f"archive {len(r['archive_F'])}")
        if si == 0:
            pareto = r["archive_F"]

        if ppo_mod.HAS_TORCH:
            t0 = time.time()
            plan_ppo, ep_r = ppo_mod.evaluate(
                V2GEnv(cfg, sc, w, grid), agent)
            plans["PPO (ours)"] = plan_ppo
            print(f"  [PPO] eval reward {ep_r:.1f} ({time.time()-t0:.1f}s)")

        for name, plan in plans.items():
            m, curves = evaluate_plan(cfg, sc, w, grid, plan, sim=sim)
            m["scenario"] = si
            all_metrics[name].append(m)
            if si == 0:
                curves0[name] = curves
        print(f"  metrics: " + " | ".join(
            f"{n}: peak {all_metrics[n][-1]['peak_kw']:.0f}" for n in plans))

    # ---------- 汇总 ----------
    rows = []
    for name in METHODS:
        df = pd.DataFrame(all_metrics[name])
        mean = df.mean(numeric_only=True)
        std = df.std(numeric_only=True)
        for k in df.columns:
            if k == "scenario":
                continue
            rows.append({"method": name, "metric": k,
                         "mean": mean[k], "std": std[k]})
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(RESULTS, "metrics_summary.csv"), index=False)
    with open(os.path.join(RESULTS, "metrics_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({m: all_metrics[m] for m in METHODS}, f,
                  ensure_ascii=False, indent=1, default=float)

    # 曲线数据（场景 0）供大屏使用
    np.savez_compressed(
        os.path.join(RESULTS, "curves.npz"),
        base_load=Grid33(cfg).base_P.sum(axis=0),
        price=cfg.price_array(),
        **{f"load_{n}": curves0[n]["total_load"] for n in curves0},
        **{f"vmin_{n}": curves0[n]["v_min"] for n in curves0},
        **{f"pst_{n}": curves0[n]["p_station"] for n in curves0},
        **{f"soc_{n}": curves0[n]["soc"] for n in curves0},
    )

    make_figures(cfg, curves0, pareto, train_hist, df_summary, w0)
    print(f"\nDone in {time.time()-t_start:.0f}s. Results -> {RESULTS}")
    return df_summary


# ----------------------------------------------------------------------
def _pivot(df, metric):
    sub = df[df.metric == metric]
    return (sub.set_index("method")[["mean", "std"]]
            .reindex([m for m in METHODS if m in sub.method.values]))


def make_figures(cfg, curves0, pareto, train_hist, df_summary, w0=None):
    T = cfg.T
    th = np.arange(T) * cfg.DT
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3})

    # ---- 1. 负荷曲线 ----
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(th, np.load(os.path.join(RESULTS, "curves.npz"))["base_load"],
            "k--", lw=1.2, label="Base load")
    for n in METHODS:
        if n in curves0:
            ax.plot(th, curves0[n]["total_load"], color=COLORS[n],
                    lw=1.6, label=n)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Total load (kW)")
    ax.set_title("System load profile under different strategies")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2)); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_load_curves.png"))
    plt.close(fig)

    # ---- 2. 最低电压 ----
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.axhline(0.95, color="gray", ls=":", lw=1)
    for n in METHODS:
        if n in curves0:
            ax.plot(th, curves0[n]["v_min"], color=COLORS[n], lw=1.5, label=n)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Minimum bus voltage (p.u.)")
    ax.set_title("Feeder minimum voltage profile")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2)); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_voltage.png"))
    plt.close(fig)

    # ---- 3. MOPSO Pareto 前沿 ----
    if pareto is not None:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        pairs = [(0, 1, "f1 (grid)", "f2 (user)"),
                 (0, 2, "f1 (grid)", "f3 (station)"),
                 (1, 2, "f2 (user)", "f3 (station)")]
        for ax, (i, j, xl, yl) in zip(axes, pairs):
            ax.scatter(pareto[:, i], pareto[:, j], s=14, c="#1f77b4", alpha=0.7)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
        fig.suptitle("MOPSO Pareto front (normalized objectives)")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "fig_pareto.png")); plt.close(fig)
        pd.DataFrame(pareto, columns=["f1_grid", "f2_user", "f3_station"]) \
            .to_csv(os.path.join(RESULTS, "pareto.csv"), index=False)

    # ---- 4. PPO 训练曲线 ----
    if train_hist:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
        r = np.array(train_hist["reward"])
        v = np.array(train_hist["viol"])
        k = max(len(r) // 30, 1)
        smooth = np.convolve(r, np.ones(k) / k, mode="valid")
        axes[0].plot(r, alpha=0.25, color="#2ca02c")
        axes[0].plot(np.arange(k - 1, len(r)), smooth, color="#2ca02c", lw=1.6)
        axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Episode reward")
        axes[0].set_title("PPO training reward")
        vs = np.convolve(v, np.ones(k) / k, mode="valid")
        axes[1].plot(np.arange(k - 1, len(v)), vs, color="#d62728", lw=1.6)
        axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Violations / episode")
        axes[1].set_title("Departure-SOC violations (smoothed)")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "fig_training.png")); plt.close(fig)
        pd.DataFrame({"reward": r, "viol": v}).to_csv(
            os.path.join(RESULTS, "ppo_rewards.csv"), index=False)

    # ---- 5. 关键指标条形图 ----
    metrics_show = [("pv_diff_kw", "Peak-valley diff (kW)"),
                    ("loss_kwh", "Daily losses (kWh)"),
                    ("v_dev_pu_h", "Voltage deviation (p.u.h)"),
                    ("user_cost_mean", "Mean user cost (yuan)"),
                    ("station_rev", "Station revenue (yuan)"),
                    ("v2g_energy_kwh", "V2G energy (kWh)")]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2))
    for ax, (mkey, mlabel) in zip(axes.ravel(), metrics_show):
        pv = _pivot(df_summary, mkey)
        ax.bar(range(len(pv)), pv["mean"], yerr=pv["std"].fillna(0),
               color=[COLORS[m] for m in pv.index], capsize=3)
        ax.set_xticks(range(len(pv)))
        ax.set_xticklabels([m.replace(" (ours)", "\n(ours)") for m in pv.index],
                           fontsize=7)
        ax.set_title(mlabel, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_metrics_bar.png")); plt.close(fig)

    # ---- 6. 站点功率 & SOC 轨迹（PPO）----
    if "PPO (ours)" in curves0:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        pst = curves0["PPO (ours)"]["p_station"]
        ax = axes[0]
        ax.stackplot(th, np.where(pst > 0, pst, 0),
                     colors=plt.cm.Blues(np.linspace(0.35, 0.75, len(pst))))
        ax.plot(th, -np.where(pst < 0, -pst, 0).sum(axis=0), color="#d62728",
                lw=1.4, label="V2G discharge")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Station power (kW)")
        ax.set_title("PPO dispatch: station-level power"); ax.legend(fontsize=8)
        ax = axes[1]
        soc = curves0["PPO (ours)"]["soc"][:30]
        im = ax.imshow(soc, aspect="auto", origin="lower", cmap="viridis",
                       extent=[0, 24, 0, 30], vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="SOC")
        ax.set_xlabel("Hour of day"); ax.set_ylabel("EV index")
        ax.set_title("SOC trajectories (first 30 EVs, PPO)")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "fig_soc_power.png")); plt.close(fig)

    # ---- 7. 意愿度分布 ----
    if w0 is not None:
        fig, ax = plt.subplots(figsize=(6.5, 3))
        ax.hist(w0.ravel(), bins=40, color="#1f77b4", alpha=0.85)
        ax.axvline(cfg.W_DISCHARGE, color="#d62728", ls="--", lw=1.2,
                   label=f"Discharge threshold = {cfg.W_DISCHARGE}")
        ax.set_xlabel("Fuzzy willingness degree")
        ax.set_ylabel("Count (EV, slot)")
        ax.set_title("Distribution of V2G participation willingness")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "fig_willingness.png"))
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速冒烟模式")
    ap.add_argument("--scen", type=int, default=None, help="场景数量")
    args = ap.parse_args()
    cfg = Config()
    n = args.scen or cfg.N_SCENARIO
    df = run(cfg, n, quick=args.quick)
    piv = df.pivot_table(index="metric", columns="method", values="mean")
    pd.set_option("display.width", 200)
    print("\n===== METRICS (mean over scenarios) =====")
    print(piv.round(3).to_string())
