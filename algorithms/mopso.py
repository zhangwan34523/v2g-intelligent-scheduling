# -*- coding: utf-8 -*-
"""策略三：改进多目标粒子群优化（MOPSO，论文方法的工程化实现）。

问题：决策变量为 6 站 × 96 时段的站点功率指令 X ∈ [-1,1]^{NS×T}，
解码为站点目标功率（>0 充电、<0 放电，幅值 × 站点额定容量），
下层由仿真内核分配到车并自动满足个体约束。

目标（与论文一致的三方利益）：
  f1 电网运行成本 = 网损成本 + 负荷波动惩罚
  f2 用户综合成本 = 充电电费 + 电池退化 − V2G 补偿
  f3 电站收益取负 = −(充电服务毛利 + V2G 售电收益 − 补偿支出)

改进点（相对标准 MOPSO）：
  1. 惯性权重线性递减（0.9 → 0.4）平衡全局/局部搜索
  2. 以概率 PSO_TURB 对粒子位进行高斯扰动（变异），增强多样性
  3. 外部档案 + 拥挤距离维护 Pareto 前沿，gbest 按拥挤距离轮盘选择
  4. 模糊隶属度法选择最终折衷解
"""

import numpy as np

from v2g.metrics import evaluate_plan
from v2g.environment import V2GSim
from algorithms.uncoordinated import uncoordinated_plan


class MOPSO:
    def __init__(self, cfg, scenario, w_matrix, grid, seed=0,
                 pop=None, iters=None, verbose=True):
        self.cfg = cfg
        self.sc = scenario
        self.sim = V2GSim(cfg, scenario, w_matrix)
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        self.pop = pop or cfg.PSO_POP
        self.iters = iters or cfg.PSO_ITER
        self.verbose = verbose
        self.NS, self.T = self.sim.NS, cfg.T
        self.D = self.NS * self.T
        # 站点功率解码上限（静态容量，实际受 SOC 约束自动截断）
        n_max = max(scenario.station_counts().max(axis=1).max(), 10)
        self.p_st_max = n_max * cfg.P_CH_MAX        # 站点充电容量上限 kW
        self.p_dis_st_max = n_max * cfg.P_DIS_MAX * 0.8
        # 惯例：网损/波动基准（用于目标归一化），由无序充电基线估计
        base_m, _ = evaluate_plan(cfg, scenario, w_matrix, grid,
                                  uncoordinated_plan(cfg, self.sim))
        self.f1_base = max(base_m["network_cost"], 1.0)
        self.f2_base = max(base_m["user_cost_total"], 1.0)
        self.f3_base = max(abs(base_m["station_rev"]), 1.0)

    # ------------------------------------------------------------------
    def _decode(self, X):
        """(D,) 或 (pop, D) → 站点功率计划 (NS, T) 或 (pop, NS, T)。"""
        single = (X.ndim == 1)
        X2 = np.atleast_2d(X).reshape(-1, self.NS, self.T)
        P = np.where(X2 > 0, X2 * self.p_st_max, X2 * self.p_dis_st_max)
        return P[0] if single else P

    def _objectives(self, P):
        """功率计划 (NS,T) → 三目标。"""
        m, _ = evaluate_plan(self.cfg, self.sc, self.sim.w, self.grid, P,
                             sim=self.sim)
        f1 = m["network_cost"] / self.f1_base
        f2 = m["user_cost_total"] / self.f2_base
        f3 = -m["station_rev"] / self.f3_base
        # 违约硬惩罚：每辆离场 SOC 不达标的车计入显著罚项（归一化目标 ~1.0 量级）
        f2 += 10.0 * m["violations"]
        return np.array([f1, f2, f3]), m

    # ------------------------------------------------------------------
    def _nondominated(self, F):
        """返回非支配解掩码。"""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            worse_or_equal = np.all(F <= F[i] + 1e-12, axis=1) & \
                             np.any(F < F[i] - 1e-12, axis=1)
            if np.any(worse_or_equal):
                dominated[i] = True
                continue
            # 互相支配检查：i 是否被 j 支配
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i] + 1e-12) and np.any(F[j] < F[i] - 1e-12):
                    dominated[i] = True
                    break
        return ~dominated

    def _crowding(self, F):
        """拥挤距离（越大越稀疏）。"""
        n, k = F.shape
        dist = np.zeros(n)
        for j in range(k):
            order = np.argsort(F[:, j])
            f_sorted = F[order, j]
            fmin, fmax = f_sorted[0], f_sorted[-1]
            if fmax - fmin < 1e-12:
                continue
            dist[order[0]] = dist[order[-1]] = np.inf
            norm = fmax - fmin
            dist[order[1:-1]] += (f_sorted[2:] - f_sorted[:-2]) / norm
        return dist

    # ------------------------------------------------------------------
    def solve(self):
        cfg = self.cfg
        rng = self.rng
        lb, ub = -1.0, 1.0

        X = rng.uniform(lb, ub, (self.pop, self.D))
        V = np.zeros_like(X)
        F = np.zeros((self.pop, 3))
        M = [None] * self.pop
        for i in range(self.pop):
            F[i], M[i] = self._objectives(self._decode(X[i]))

        # 初始档案
        nd = self._nondominated(F)
        archive_X, archive_F = X[nd].copy(), F[nd].copy()
        pbest_X, pbest_F = X.copy(), F.copy()

        hist = np.zeros(self.iters)
        for it in range(self.iters):
            w = 0.9 - 0.5 * it / max(self.iters - 1, 1)
            for i in range(self.pop):
                # gbest：从档案按拥挤距离轮盘选（偏好稀疏区域）
                if len(archive_F) > 1:
                    cd = self._crowding(archive_F)
                    finite = cd[np.isfinite(cd)]
                    cap = finite.max() * 2.0 if len(finite) else 1.0
                    cd = np.where(np.isfinite(cd), cd, cap)
                    p = cd / cd.sum()
                    g = archive_X[rng.choice(len(archive_X), p=p)]
                else:
                    g = archive_X[0]
                r1, r2 = rng.random(self.D), rng.random(self.D)
                V[i] = (w * V[i] + cfg.PSO_C1 * r1 * (pbest_X[i] - X[i])
                        + cfg.PSO_C2 * r2 * (g - X[i]))
                V[i] = np.clip(V[i], -0.3, 0.3)
                X[i] = np.clip(X[i] + V[i], lb, ub)
                # 变异扰动（改进项）
                turb = rng.random(self.D) < cfg.PSO_TURB
                X[i][turb] += rng.normal(0, 0.1, turb.sum())
                X[i] = np.clip(X[i], lb, ub)

                F[i], M[i] = self._objectives(self._decode(X[i]))
                # 个人最优（Pareto 支配关系；互不支配时随机替换）
                if _dominates(F[i], pbest_F[i]):
                    pbest_X[i], pbest_F[i] = X[i].copy(), F[i].copy()
                elif not _dominates(pbest_F[i], F[i]) and rng.random() < 0.5:
                    pbest_X[i], pbest_F[i] = X[i].copy(), F[i].copy()

            # 更新档案
            cand = np.vstack([archive_X, X])
            candF = np.vstack([archive_F, F])
            nd = self._nondominated(candF)
            archive_X, archive_F = cand[nd], candF[nd]
            if len(archive_X) > cfg.PSO_ARCHIVE:
                cd = self._crowding(archive_F)
                keep = np.argsort(-cd)[:cfg.PSO_ARCHIVE]
                archive_X, archive_F = archive_X[keep], archive_F[keep]

            hist[it] = archive_F.mean()
            if self.verbose and (it + 1) % 20 == 0:
                print(f"  [MOPSO] iter {it+1}/{self.iters} "
                      f"archive={len(archive_F)} mean_f={hist[it]:.3f}")

        # ---- 模糊隶属度折衷解 ----
        fmin, fmax = archive_F.min(0), archive_F.max(0)
        span = np.maximum(fmax - fmin, 1e-12)
        mu = (fmax[None, :] - archive_F) / span[None, :]
        knee = int(np.argmax(mu.min(axis=1)))
        best_X = archive_X[knee]
        best_plan = self._decode(best_X)

        # ---- 安全部投影（两遍法）----
        # 第一遍：沿折衷解的 SOC 轨迹计算各时段"最低充电需求"，
        # 第二遍：最终计划取 max(优化解, 安全基线)，保证离场零违约。
        res1 = self.sim.run(best_plan)
        base = np.zeros((self.NS, self.T))
        soc_traj = res1["soc"]
        for t in range(self.T):
            base[:, t] = self.sim.base_charge_target(soc_traj[:, t], t)
        best_plan = np.maximum(best_plan, base)

        return {"plan": best_plan, "archive_F": archive_F,
                "archive_X": archive_X, "knee_idx": knee, "hist": hist}


def _dominates(a, b):
    return np.all(a <= b + 1e-12) and np.any(a < b - 1e-12)
