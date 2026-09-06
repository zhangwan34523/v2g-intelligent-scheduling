# -*- coding: utf-8 -*-
"""调度方案评估指标。

所有方法输出的站点级功率计划 (NS, T) 都经过同一评估流程：
  仿真内核逐时段分配 → 电网潮流 → 指标计算，
确保横向对比口径一致。
"""

import numpy as np


def evaluate_plan(cfg, scenario, w_matrix, grid, p_station, sim=None):
    """评估一个站点级功率计划。

    返回指标 dict（含全部中间曲线，供绘图）。
    """
    if sim is None:
        from .environment import V2GSim
        sim = V2GSim(cfg, scenario, w_matrix)
    res = sim.run(p_station)

    p_ev = res["p_ev"]                    # (N, T)
    p_st = res["p_station"]               # (NS, T)
    ev_bus = np.zeros((grid.NB, cfg.T))
    for s, bus in enumerate(cfg.STATION_BUS):
        ev_bus[bus] = p_st[s]

    volt, loss = grid.powerflow(ev_bus)   # (NB,T), (T,)
    total_load = grid.base_P.sum(axis=0) + p_st.sum(axis=0)   # (T,) kW

    price = cfg.price_array()
    ch_p = np.where(p_ev > 0, p_ev, 0.0)
    dis_p = np.where(p_ev < 0, -p_ev, 0.0)
    ch_e = ch_p.sum(axis=1) * cfg.DT                  # (N,) kWh
    dis_e = dis_p.sum(axis=1) * cfg.DT

    # ---- 用户侧 ----
    ch_cost = (ch_p * price[None, :] * cfg.CHARGE_SERVICE_RATE * cfg.DT).sum(axis=1)
    degr = (ch_e + dis_e) * cfg.DEGRADATION_COST
    comp = dis_e * cfg.V2G_COMPENSATION
    user_cost = ch_cost + degr - comp                 # (N,) 元

    # ---- 电站侧 ----
    st_ch_e = np.where(p_st > 0, p_st, 0.0).sum(axis=1) * cfg.DT
    st_dis_e = np.where(p_st < 0, -p_st, 0.0).sum(axis=1) * cfg.DT
    # 分时电量（按车侧口径近似电价构成）
    ch_e_by_t = (np.where(p_ev > 0, p_ev, 0.0) * cfg.DT).sum(axis=0)   # (T,)
    dis_e_by_t = (np.where(p_ev < 0, -p_ev, 0.0) * cfg.DT).sum(axis=0)
    ch_income = float((ch_e_by_t * price * cfg.CHARGE_SERVICE_RATE).sum())
    ch_purchase = float((ch_e_by_t * price).sum())
    dis_income = float((dis_e_by_t * price * cfg.GRID_BUY_RATE).sum())
    comp_pay = float(dis_e_by_t.sum() * cfg.V2G_COMPENSATION)
    station_rev = ch_income - ch_purchase + dis_income - comp_pay

    # ---- 电网侧 ----
    loss_kwh = float(loss.sum() * cfg.DT)
    v_min = volt.min(axis=0)                          # (T,)
    v_dev = float(np.clip(1.0 - cfg.V_LIMIT - v_min, 0.0, None).sum() * cfg.DT)
    peak = float(total_load.max())
    pv_diff = float(total_load.max() - total_load.min())
    load_var = float(total_load.var())
    peak_shaving = float((grid.base_P.sum(axis=0)).max() - total_load.max())

    n_viol = res["n_viol"]
    meet_rate = 1.0 - n_viol / scenario.n_ev

    m = {
        "peak_kw": peak,
        "pv_diff_kw": pv_diff,
        "load_var": load_var,
        "loss_kwh": loss_kwh,
        "v_dev_pu_h": v_dev,
        "v_min_pu": float(v_min.min()),
        "user_cost_total": float(user_cost.sum()),
        "user_cost_mean": float(user_cost.mean()),
        "station_rev": station_rev,
        "v2g_energy_kwh": float(dis_e.sum()),
        "charge_energy_kwh": float(ch_e.sum()),
        "violations": n_viol,
        "meet_rate": meet_rate,
        "network_cost": loss_kwh * cfg.LOSS_COST
                        + load_var / 1e3 * cfg.W_NETWORK,
    }
    curves = {"total_load": total_load, "v_min": v_min,
              "soc": res["soc"], "p_ev": p_ev, "p_station": p_st}
    return m, curves
