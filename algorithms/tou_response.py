# -*- coding: utf-8 -*-
"""策略二：分时电价响应（规则式基线）。

启发式规则（对应现实中"峰谷电价 + 桩端定时"的粗粒度响应）：
  · 谷段(23-7h)      ：全功率充电（填谷）
  · 平段(7-10/15-18h)：紧迫车辆充电，其余半功率缓充
  · 峰段(10-15/18-23h)：高意愿车辆放电削峰（下限受放电底线保护），
    紧迫车辆仍允许充电（防止离场违约）
不感知电网潮流，仅作电价信号响应，用于对照优化方法的价值。
"""

import numpy as np


def tou_plan(cfg, sim, scenario):
    NS, T = sim.NS, cfg.T
    price = cfg.price_array()
    plan = np.zeros((NS, T))
    for t in range(T):
        p_ch, p_dis, av = sim.power_bounds(np.full(sim.N, 0.5), t)
        # 实际充电上限与放电能力需用真实 SOC 边界：此处仅作计划，
        # 仿真时由 allocate 按 SOC 自动截断
        for s in range(NS):
            m = (sim.station_of[:, t] == s) & av
            n = int(m.sum())
            if n == 0:
                continue
            p = price[t]
            if p <= cfg.PRICE_VALLEY + 1e-9:            # 谷段
                plan[s, t] = n * cfg.P_CH_MAX
            elif p >= cfg.PRICE_PEAK - 1e-9:            # 峰段
                plan[s, t] = -0.7 * p_dis[m].sum()       # 有能力则放电削峰
            else:                                        # 平段
                plan[s, t] = 0.5 * n * cfg.P_CH_MAX
    return plan
