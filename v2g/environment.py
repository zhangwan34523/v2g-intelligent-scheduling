# -*- coding: utf-8 -*-
"""V2G 调度仿真内核与强化学习环境。

两层结构：
  上层：站点级功率决策  P_station ∈ (NS, T)，kW（充电为正/放电为负）
  下层：站内单车功率分配（充电按紧迫度、放电按意愿度排序，受 SOC 与
        离场需求约束裁剪）

所有调度算法（无序充电、分时电价响应、MOPSO、PPO）共用同一仿真内核，
且在同一场景（公共随机数 CRN）下运行，保证对比公平。

事件时序（每时段 t）：
  1) 到站事件：白天窗口/夜间窗口到达，SOC 扣减对应行驶能耗
  2) 上层站点功率决策 → 下层单车分配
  3) SOC 演化（充/放效率、SOC 上下限）
  4) 离场事件：今晨离场需满足 SOC 需求（违约计罚）；白天窗口离场
     需 SOC ≥ 0.35（保证可返家，由放电底线事前防护）
"""

import numpy as np


class V2GSim:
    """全天批量仿真器：给定站点功率计划，返回完整调度结果。"""

    def __init__(self, cfg, scenario, w_matrix):
        self.cfg = cfg
        self.sc = scenario
        self.w = w_matrix                      # (N, T) 意愿度
        self.N = scenario.n_ev
        self.T = cfg.T
        self.NS = len(cfg.STATION_BUS)
        self.e_cap = scenario.e_cap
        self.soc_need = scenario.soc_need
        self.avail = scenario.avail
        self.station_of = scenario.station_of
        self.has_day = scenario.has_day
        # 离场/到站时段索引
        self.t_out_slot = np.ceil(scenario.t_out1 * 4).astype(int)
        self.t_in2_slot = np.floor(scenario.t_in2 * 4).astype(int)
        self.t_out2_slot = np.ceil(scenario.t_out2 * 4).astype(int)
        self.t_in3_slot = np.floor(scenario.t_in3 * 4).astype(int)
        self.t_in1_slot = np.floor(scenario.t_in1 * 4).astype(int)

    # ------------------------------------------------------------------
    def power_bounds(self, soc, t):
        """(N,) 时段 t 各车充/放电功率上下限（kW）与可用掩码。

        放电底线（防违约的事前约束）：
          · 早间离场临近且剩余时间不足以补回 SOC 需求 → 底线抬至 soc_need
          · 白天窗口离场前 → 底线不低于 0.35（返家保底）
          · 其余 → SOC_MIN
        """
        cfg = self.cfg
        av = self.avail[:, t]
        p_ch = np.where(av & (soc < cfg.SOC_MAX - 1e-9), cfg.P_CH_MAX, 0.0)

        remain_h = np.maximum(self.t_out_slot - t, 0.0) * cfg.DT
        need_reach_h = (np.maximum(self.soc_need - soc, 0.0) * self.e_cap
                        / cfg.P_CH_MAX / cfg.ETA_CH)
        tight = (t < self.t_out_slot) & (remain_h < need_reach_h * 1.10)
        floor = np.where(tight, self.soc_need, cfg.SOC_MIN)
        # 白天窗口期间底线 0.35
        day_phase = self.has_day & (t >= self.t_out_slot) & (t < self.t_out2_slot)
        floor = np.where(day_phase, np.maximum(floor, 0.35), floor)

        p_dis_cap = self.w[:, t] * cfg.P_DIS_MAX
        can_dis = av & (soc > floor + 1e-9) & (p_dis_cap > 0.0)
        room_kwh = np.maximum((soc - floor) * self.e_cap, 0.0)
        p_dis_max = np.where(can_dis,
                             np.minimum(p_dis_cap, room_kwh / cfg.DT * cfg.ETA_DIS),
                             0.0)
        return p_ch, p_dis_max, av

    # ------------------------------------------------------------------
    def allocate_station(self, soc, t, s, p_target, bounds=None):
        """单站目标功率 → 单车功率分配（向量化贪心）。

        充电：紧迫度（亏缺能量/剩余时间）降序依次满足；
        放电：意愿度降序依次满足。
        排序后用 cumsum 前缀和一次性完成"依次填充"，与逐车贪心等价。
        bounds: 可选预计算的 (p_ch, p_dis, av)，避免跨站重复计算。
        返回 (p_ev (N,), 实际站点功率 kW)
        """
        cfg = self.cfg
        p_ev = np.zeros(self.N)
        in_st = (self.station_of[:, t] == s)
        if not in_st.any() or abs(p_target) < 1e-6:
            return p_ev, 0.0
        p_ch, p_dis, av = bounds if bounds is not None \
            else self.power_bounds(soc, t)

        if p_target > 0:                                   # ---- 充电 ----
            idx = np.where(in_st & av & (p_ch > 0))[0]
            if len(idx) == 0:
                return p_ev, 0.0
            remain_h = np.maximum(self.t_out_slot[idx] - t, 0.5) * cfg.DT
            urg = (np.maximum(self.soc_need[idx] - soc[idx], 0.0)
                   * self.e_cap[idx]) / remain_h / cfg.ETA_CH
            order = idx[np.argsort(-urg)]
            cap = p_ch[order]
            cum_excl = np.concatenate([[0.0], np.cumsum(cap)[:-1]])
            alloc = np.clip(np.minimum(cap, p_target - cum_excl), 0.0, cap)
            p_ev[order] = alloc
            return p_ev, float(alloc.sum())     # 实际分配功率（kW）
        else:                                              # ---- 放电 ----
            idx = np.where(in_st & av & (p_dis > 0))[0]
            if len(idx) == 0:
                return p_ev, 0.0
            order = idx[np.argsort(-self.w[idx, t])]
            cap = p_dis[order]
            need = -p_target
            cum_excl = np.concatenate([[0.0], np.cumsum(cap)[:-1]])
            alloc = np.clip(np.minimum(cap, need - cum_excl), 0.0, cap)
            p_ev[order] = -alloc
            return p_ev, -float(alloc.sum())    # 实际放电功率（负值，kW）

    # ------------------------------------------------------------------
    def base_charge_target(self, soc, t):
        """安全基线的站点充电目标功率 (NS,)。

        对"离场时刻在仿真视界内且尚未离场"的车辆，按
          p_req_i = SOC 亏缺能量 / (剩余在站时间 × 充电效率) × 1.2 安全系数
        计算恰好满足离场需求的最低充电功率并按站求和（上限额定功率）。
        该基线保证 mu=0 的确定性策略也不发生离场违约；
        RL 在此之上学习经济性最优的偏离（残差策略学习）。
        """
        cfg = self.cfg
        target = np.zeros(self.NS)
        in_horizon = self.t_out_slot <= self.T
        for s in range(self.NS):
            mask = (self.station_of[:, t] == s) & self.avail[:, t] \
                & in_horizon & (t < self.t_out_slot)
            if not mask.any():
                continue
            deficit_kwh = np.maximum(self.soc_need[mask] - soc[mask], 0.0) \
                * self.e_cap[mask]
            remain_h = np.maximum(self.t_out_slot[mask] - t, 0.25) * cfg.DT
            p_req = deficit_kwh / (remain_h * cfg.ETA_CH) * 1.2
            target[s] = np.minimum(p_req, cfg.P_CH_MAX).sum()
        return target

    # ------------------------------------------------------------------
    def step(self, soc, t, p_station_t):
        """推进一个时段：分配 + SOC 演化 + 事件处理。soc 原地更新。

        参数 p_station_t : (NS,) 站点目标功率
        返回 (p_ev (N,), p_st_actual (NS,))
        """
        cfg = self.cfg
        # 1) 到站事件（SOC 扣减行驶能耗）
        ev_in2 = (t == self.t_in2_slot) & self.has_day
        ev_in3 = (t == self.t_in3_slot) & self.has_day
        ev_in1 = (t == self.t_in1_slot) & (~self.has_day)   # 无白天窗口车傍晚返家
        if ev_in2.any():
            soc[ev_in2] = np.maximum(
                soc[ev_in2] - self.sc.e_drive1[ev_in2] / self.e_cap[ev_in2],
                cfg.SOC_MIN)
        if ev_in3.any():
            soc[ev_in3] = np.maximum(
                soc[ev_in3] - self.sc.e_drive2[ev_in3] / self.e_cap[ev_in3],
                cfg.SOC_MIN)
        if ev_in1.any():
            soc[ev_in1] = self.sc.soc0[ev_in1]              # 模式重复：回到达 SOC

        # 2) 逐站分配（功率边界只算一次，复用于 6 个站）
        p_ev = np.zeros(self.N)
        p_st = np.zeros(self.NS)
        soc_snap = soc.copy()
        bounds = self.power_bounds(soc_snap, t)
        for s in range(self.NS):
            pev, pst = self.allocate_station(soc_snap, t, s,
                                             p_station_t[s], bounds)
            p_ev += pev
            p_st[s] = pst

        # 3) SOC 演化
        ch = p_ev > 0
        soc = np.where(ch, np.minimum(
            soc + p_ev * cfg.DT * cfg.ETA_CH / self.e_cap, cfg.SOC_MAX), soc)
        dis = p_ev < 0
        soc = np.where(dis, np.maximum(
            soc + p_ev * cfg.DT / (cfg.ETA_DIS * self.e_cap), cfg.SOC_MIN), soc)

        # 4) 离场校验（今晨离场 SOC 需求）
        departed = (t + 1 == self.t_out_slot)
        n_viol = int((departed & (soc < self.soc_need - 1e-3)).sum())
        return p_ev, p_st, soc, n_viol

    # ------------------------------------------------------------------
    def run(self, p_station):
        """全天批量仿真。p_station: (NS, T)。

        返回 dict(soc, p_ev, p_station_actual, violations, n_viol)
        """
        N, T, NS = self.N, self.T, self.NS
        soc = self.sc.soc0.copy()
        soc_traj = np.zeros((N, T))
        p_ev_traj = np.zeros((N, T))
        p_st_traj = np.zeros((NS, T))
        n_viol = 0
        for t in range(T):
            soc_traj[:, t] = soc
            p_ev, p_st, soc, nv = self.step(soc, t, p_station[:, t])
            p_ev_traj[:, t] = p_ev
            p_st_traj[:, t] = p_st
            n_viol += nv
        return {"soc": soc_traj, "p_ev": p_ev_traj,
                "p_station": p_st_traj, "n_viol": n_viol}


class V2GEnv:
    """Gym 风格逐步调度环境（供 PPO 训练/评估）。

    状态（39 维）：全局 9 维 + 每站 5 维 × 6 站
    动作：(NS,) ∈ [-1,1]^6；>0 充电、<0 放电，幅值为可调容量比例
    奖励：−(网损 + 电压越限 + 用户成本 − 电站收益) − 离场违约惩罚
    """

    def __init__(self, cfg, scenario, w_matrix, grid,
                 reward_weights=(1.0, 1.0, 1.0, 1.0)):
        self.cfg = cfg
        self.sc = scenario
        self.grid = grid
        self.sim = V2GSim(cfg, scenario, w_matrix)
        self.N, self.T, self.NS = self.sim.N, cfg.T, self.sim.NS
        self.wL, self.wV, self.wU, self.wS = reward_weights
        self.shape_k = getattr(cfg, "PPO_SHAPE_K", 0.02)
        self.price = cfg.price_array()
        self.base_norm = grid.base_P.sum(axis=0) / grid.base_P.sum(axis=0).max()
        self.st_of = scenario.station_of
        self.action_dim = self.NS
        self.obs_dim = 9 + 5 * self.NS
        self.reset()

    # ------------------------------------------------------------------
    def _obs(self, t):
        cfg = self.cfg
        av = self.sim.avail[:, t]
        st = self.st_of[:, t]
        p_ch, p_dis, _ = self.sim.power_bounds(self.soc, t)
        feat = np.zeros(5 * self.NS)
        for s in range(self.NS):
            m = av & (st == s)
            n_ev = int(m.sum())
            if n_ev:
                mean_soc = float(self.soc[m].mean())
                deficit = float((np.maximum(self.sim.soc_need[m] - self.soc[m], 0)
                                 * self.sim.e_cap[m]).sum()) / 1e3
            else:
                mean_soc, deficit = 0.0, 0.0
            head = float(p_ch[m & (p_ch > 0)].sum())
            foot = float(p_dis[m & (p_dis > 0)].sum())
            feat[s * 5:s * 5 + 5] = [n_ev / 25.0, mean_soc, deficit / 2e3,
                                     head / 200.0, foot / 100.0]
        h = t * cfg.DT
        g = [t / self.T,
             np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
             (self.price[t] - cfg.PRICE_VALLEY) / (cfg.PRICE_PEAK - cfg.PRICE_VALLEY),
             self.base_norm[t],
             float(av.sum()) / 100.0,
             float(self.soc[av].mean()) if av.any() else 0.0,
             float((np.maximum(self.sim.soc_need[av] - self.soc[av], 0)
                    * self.sim.e_cap[av]).sum()) / 1e3 / 2000.0 if av.any() else 0.0,
             self._last_act_mean]
        return np.concatenate([np.array(g, dtype=np.float32), feat.astype(np.float32)])

    # ------------------------------------------------------------------
    def reset(self):
        self.t = 0
        self.soc = self.sc.soc0.copy()
        self._last_act_mean = 0.0
        self._viol_total = 0
        self._log = []
        # 势函数（potential-based shaping）：聚合亏缺能量 kWh
        self._deficit = float((np.maximum(self.sim.soc_need - self.soc, 0)
                               * self.sim.e_cap).sum())
        return self._obs(0)

    # ------------------------------------------------------------------
    def step(self, action):
        cfg = self.cfg
        t = self.t
        action = np.clip(np.asarray(action, dtype=float).flatten(), -1.0, 1.0)
        self._last_act_mean = float(action.mean())

        # 残差决策：目标 = 安全基线 + a·站点充电容量（截断到可调范围）
        p_ch, p_dis, _ = self.sim.power_bounds(self.soc, t)
        base = self.sim.base_charge_target(self.soc, t)
        target = np.zeros(self.NS)
        for s in range(self.NS):
            m_s = (self.st_of[:, t] == s)
            head = float(p_ch[m_s & (p_ch > 0)].sum())
            foot = float(p_dis[m_s & (p_dis > 0)].sum())
            target[s] = np.clip(base[s] + action[s] * head, -foot, head)

        p_ev, p_st, soc_new, n_viol = self.sim.step(self.soc, t, target)
        self.soc = soc_new
        self._viol_total += n_viol

        # 电网与经济指标（单时段）
        ev_p = np.zeros((self.grid.NB, 1))
        for s in range(self.NS):
            ev_p[cfg.STATION_BUS[s], 0] = p_st[s]
        volt, loss = self.grid.powerflow(ev_p)
        v_dev = max(0.0, (1.0 - cfg.V_LIMIT) - float(volt[:, 0].min()))

        ch_e = float(p_ev[p_ev > 0].sum()) * cfg.DT
        dis_e = float(-p_ev[p_ev < 0].sum()) * cfg.DT
        user_cost = (ch_e * self.price[t] * cfg.CHARGE_SERVICE_RATE
                     + (ch_e + dis_e) * cfg.DEGRADATION_COST
                     - dis_e * cfg.V2G_COMPENSATION)
        station_rev = (ch_e * self.price[t] * (cfg.CHARGE_SERVICE_RATE - 1.0)
                       + dis_e * (self.price[t] * cfg.GRID_BUY_RATE
                                  - cfg.V2G_COMPENSATION))
        reward = -(self.wL * float(loss[0]) / 50.0
                   + self.wV * v_dev * 20.0
                   + self.wU * user_cost / 20.0
                   - self.wS * station_rev / 20.0
                   + 8.0 * n_viol)
        # 势函数整形：亏缺能量下降给正奖励（保持最优策略不变，
        # 缩短"充电动作→离场违约"之间的信用分配链）
        deficit_new = float((np.maximum(self.sim.soc_need - self.soc, 0)
                             * self.sim.e_cap).sum())
        reward += self.shape_k * (self._deficit - deficit_new)
        self._deficit = deficit_new

        self.t += 1
        done = self.t >= self.T
        obs = self._obs(min(self.t, self.T - 1))
        self._log.append({"p_station": p_st.copy(), "p_ev": p_ev.copy()})
        info = {"n_viol": n_viol, "loss": float(loss[0]),
                "user_cost": user_cost, "station_rev": station_rev}
        return obs, float(reward), done, info

    # ------------------------------------------------------------------
    def plan_matrix(self):
        """将本回合逐步决策汇总为 (NS, T) 实际站点功率（评估用）。"""
        P = np.array([rec["p_station"] for rec in self._log]).T
        return P
