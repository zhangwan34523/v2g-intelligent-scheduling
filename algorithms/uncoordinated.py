# -*- coding: utf-8 -*-
"""策略一：无序充电（基线）。

EV 接入即以额定功率充电至 SOC 上限或离场，不参与 V2G、不做任何平移，
代表"无协调管理"的现状场景。计划层面恒以满功率请求，
实际功率由仿真内核按 SOC 约束自动截断。
"""

import numpy as np


def uncoordinated_plan(cfg, sim):
    """构造无序充电的站点级功率计划 (NS, T)：在站车辆数 × 额定功率。"""
    NS, T, N = sim.NS, cfg.T, sim.N
    plan = np.zeros((NS, T))
    for t in range(T):
        for s in range(NS):
            plan[s, t] = ((sim.station_of[:, t] == s) & sim.avail[:, t]).sum() \
                * cfg.P_CH_MAX
    return plan
