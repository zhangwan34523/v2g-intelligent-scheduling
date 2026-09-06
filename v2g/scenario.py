# -*- coding: utf-8 -*-
"""基于出行链的蒙特卡洛 EV 场景生成。

参照 NHTS（美国国家家庭出行调查）与国内私家车使用统计的典型参数，
为每辆 EV 生成一日出行链：
  · 夜间住家停车（跨 0 点：昨夜 t_in1 → 今晨 t_out1 离场，离场 SOC 需求约束）
  · 白天单位/公共停车场窗口（35% 车辆，可参与工作地 V2G）
  · 夜间回住家窗口（t_in2 起，停留至仿真结束）
  · 窗口间行驶耗电量显式建模（由调度器保证离场 SOC 可达）

同一场景下所有算法使用完全相同的 EV 集合（公共随机数法 CRN），
确保对比实验的公平性与可复现性。
"""

import numpy as np


class Scenario:
    """一天内全部 EV 的场景（事件表 + 可用性掩码）。

    属性（除 evs 外均为 numpy 数组，按 EV 索引）
    ----
    n_ev          : EV 数
    avail         : (N, T) bool  各时段是否在站可调度
    station_of    : (N, T) int8  各时段所在站（-1 = 不在站）
    soc0          : (N,)   t=0 时刻 SOC（夜间窗口昨夜到达，取到站 SOC）
    soc_need      : (N,)   今晨离场前需达到的 SOC（硬约束，违约计罚）
    e_cap         : (N,)   电池容量 kWh
    t_out1        : (N,)   今晨离场时刻（小时）
    has_day       : (N,)   bool 是否有白天窗口
    e_drive1      : (N,)   今晨离场 → 白天窗口 的行驶能耗 kWh
    e_drive2      : (N,)   白天窗口 → 夜间窗口 的行驶能耗 kWh
    """

    def __init__(self, cfg, rng, n_ev=None):
        self.cfg = cfg
        self.rng = rng
        self.n_ev = n_ev or cfg.N_EV
        self._generate()

    # ------------------------------------------------------------------
    def _generate(self):
        cfg, rng, T, n = self.cfg, self.rng, self.cfg.T, self.n_ev

        # 电池容量（对数正态，截断）
        e_cap = np.clip(rng.lognormal(np.log(cfg.E_CAP_MEAN),
                                      cfg.E_CAP_SIGMA, n), 30.0, 100.0)

        # 日行驶里程 → 日耗电
        km = np.clip(rng.lognormal(np.log(cfg.KM_PER_DAY_MEAN),
                                   cfg.KM_PER_DAY_SIGMA, n), 5.0, 250.0)
        e_used = km * cfg.CONSUMPTION
        soc_arrive = np.clip(1.0 - e_used / e_cap, cfg.SOC_MIN + 0.05, cfg.SOC_MAX)

        # 今晨离场 SOC 需求
        soc_need = np.clip(rng.normal(cfg.SOC_DEPART_MEAN, cfg.SOC_DEPART_SIG, n),
                           cfg.SOC_MIN + 0.15, cfg.SOC_MAX)

        # ---- 出行链时刻（小时，[0,24)）----
        t_in1 = np.clip(rng.normal(19.0, 1.5, n), 17.0, 23.9)   # 昨夜到住家
        t_out1 = np.clip(rng.normal(7.3, 1.0, n), 5.5, 9.5)     # 今晨离场
        has_day = rng.random(n) < 0.35                          # 有白天窗口
        t_in2 = rng.uniform(9.5, 11.0, n)                       # 白天到站
        t_out2 = np.clip(t_in2 + rng.uniform(3.0, 6.0, n), 12.0, 17.5)
        t_in3 = np.clip(rng.normal(19.0, 1.5, n), 17.5, 23.9)   # 夜间回住家

        # 两段行驶能耗分配（有白天窗口的车）
        split = rng.uniform(0.3, 0.7, n)
        e_drive1 = np.where(has_day, e_used * split, 0.0)
        e_drive2 = np.where(has_day, e_used * (1 - split), 0.0)

        # ---- 站点分配 ----
        st_ratio = np.array(cfg.STATION_RATIO, dtype=float)
        st_ratio = st_ratio / st_ratio.sum()
        # 夜间窗口（住家）与白天窗口（单位/公共）可属不同站
        st_night = rng.choice(len(st_ratio), size=n, p=st_ratio)
        st_day = rng.choice(len(st_ratio), size=n, p=st_ratio)

        # ---- 时段掩码 ----
        def slot_mask(t_a, t_b):
            m = np.zeros(T, dtype=bool)
            ia = int(np.clip(np.floor(t_a * 4), 0, T))
            ib = int(np.clip(np.ceil(t_b * 4), 0, T))
            if ia <= ib:
                m[ia:ib] = True
            else:                       # 跨 0 点（夜间窗口）
                m[ia:] = True
                m[:ib] = True
            return m

        avail = np.zeros((n, T), dtype=bool)
        station_of = np.full((n, T), -1, dtype=np.int8)
        m1 = np.array([slot_mask(t_in1[i], t_out1[i]) for i in range(n)])
        m3 = np.array([slot_mask(t_in3[i], 24.0) for i in range(n)])
        avail |= m1
        avail |= m3
        station_of[m1] = st_night[:, None].repeat(T, 1)[m1]
        station_of[m3] = st_night[:, None].repeat(T, 1)[m3]
        if has_day.any():
            m2 = np.array([slot_mask(t_in2[i], t_out2[i]) if has_day[i]
                           else np.zeros(T, dtype=bool) for i in range(n)])
            avail |= m2
            station_of[m2] = st_day[:, None].repeat(T, 1)[m2]

        # ---- 装载 ----
        self.avail = avail
        self.station_of = station_of
        self.soc0 = soc_arrive.copy()
        self.soc_need = soc_need
        self.e_cap = e_cap
        self.t_in1, self.t_out1 = t_in1, t_out1
        self.t_in2, self.t_out2, self.t_in3 = t_in2, t_out2, t_in3
        self.has_day = has_day
        self.e_drive1, self.e_drive2 = e_drive1, e_drive2
        self.station_night, self.station_day = st_night, st_day
        self.t_out2 = t_out2
        self._build_departure_tables()

    # ------------------------------------------------------------------
    def _build_departure_tables(self):
        """构造 (N, T) 的"下一次离场时段"与"对应 SOC 需求"表。

        时段分段（对每辆车）：
          · 早间段   t < t_out1        ：昨夜住家窗口，离场 = t_out1，
                                        需求 = soc_need（硬约束）
          · 白天窗口 t_out1..t_out2    ：单位/公共窗口，离场 = t_out2，
                                        需求 = 0.35（返家保底）
          · 夜间段   t_in3 之后        ：住家窗口，离场 = t_out1 + T（次日晨，
                                        超出仿真视界），需求 = soc_need
        """
        cfg, T, n = self.cfg, self.cfg.T, self.n_ev
        t_axis = np.arange(T)[None, :]
        t_out_slot = np.ceil(self.t_out1 * 4).astype(int)[:, None]
        t_out2_slot = np.ceil(self.t_out2 * 4).astype(int)[:, None]
        t_in1_slot = np.floor(self.t_in1 * 4).astype(int)[:, None]
        t_in3_slot = np.floor(self.t_in3 * 4).astype(int)[:, None]

        morning = t_axis < t_out_slot
        day_phase = (self.has_day[:, None] & (t_axis >= t_out_slot)
                     & (t_axis < t_out2_slot))
        evening = np.where(self.has_day[:, None],
                           t_axis >= t_in3_slot, t_axis >= t_in1_slot)

        t_dep = np.where(morning, t_out_slot,
                         np.where(day_phase, t_out2_slot, t_out_slot + T))
        need_dep = np.where(morning, self.soc_need[:, None],
                            np.where(day_phase, 0.35, self.soc_need[:, None]))
        # 行驶间隙（不在站）置 0，避免误用
        t_dep[~self.avail] = 0
        need_dep[~self.avail] = cfg.SOC_MIN
        self.t_dep_next = t_dep
        self.need_at_dep = need_dep

    # ------------------------------------------------------------------
    def station_counts(self):
        """(NS, T) 每时段各站接入 EV 数（用于掩码与可视化）。"""
        NS, T = len(self.cfg.STATION_BUS), self.cfg.T
        cnt = np.zeros((NS, T))
        for s in range(NS):
            cnt[s] = ((self.station_of == s) & self.avail).sum(axis=0)
        return cnt

    def summary(self):
        return {
            "n_ev": self.n_ev,
            "with_day_window": int(self.has_day.sum()),
            "mean_e_cap": float(self.e_cap.mean()),
            "mean_soc0": float(self.soc0.mean()),
            "mean_soc_need": float(self.soc_need.mean()),
        }
