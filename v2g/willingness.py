# -*- coding: utf-8 -*-
"""用户 V2G 参与意愿的模糊逻辑量化模型。

延续本人 ACPEE 2026 论文中的建模思路：以"电池退化焦虑"与"时间焦虑"
为核心，用 Mamdani 型模糊推理将多维用户体验因素映射为连续意愿度 w∈[0,1]，
并以意愿度作为可调度放电容量系数（w 越高，允许 V2G 放电功率越大），
从而把"用户是否愿意"这一软约束显式嵌入优化与强化学习调度。

输入（归一化）：
  x1 SOC 裕度   ：(SOC − SOC_need)/(SOC_max − SOC_need)，反映退化焦虑
  x2 停留裕度   ：剩余在站时长超出"以额定功率补足 SOC 需求"所需时长的
                  相对裕度，反映时间焦虑
  x3 电价水平   ：分时电价线性归一化（谷=0，峰=1），反映补偿激励

输出：意愿度 w（单点输出 + 强度加权解模糊）。
决策规则：w ≥ W_DISCHARGE 才允许放电，且放电上限 = w·P_dis；
未达 W_DISCHARGE 的用户仅接受充电平移，不接受放电。
"""

import numpy as np


def _tri(x, a, b, c):
    """三角/肩形隶属函数（支持 a==b 左肩、b==c 右肩的退化形式）。

    退化情形必须显式处理：右肩形在 x≥c 处隶属恒为 1，
    若用 (c−x)/1e-9 数值计算会在峰值点得到 0（经典错误）。
    """
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a) if b > a else np.where(x <= a, 1.0, 0.0)
    right = (c - x) / (c - b) if c > b else np.where(x >= c, 1.0, 0.0)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def _mfs(x):
    """将 [0,1] 输入映射为 (低, 中, 高) 三个隶属度。

    低：左肩形（x=0 时为 1）；中：三角；高：右肩形（x=1 时为 1）。
    """
    lo = _tri(x, 0.0, 0.0, 0.5)
    mid = _tri(x, 0.0, 0.5, 1.0)
    hi = _tri(x, 0.5, 1.0, 1.0)
    return lo, mid, hi


def willingness(soc, need_at_dep, e_cap, t_dep_next, price_arr, cfg):
    """计算各 EV 在各时段的 V2G 参与意愿度。

    参数
    ----
    soc        : (N, T) 各时段 SOC（离线量化时取"立即充电"参考轨迹）
    need_at_dep: (N, T) 各时段对应下一次离场的 SOC 需求
    e_cap      : (N,)   电池容量 kWh
    t_dep_next : (N, T) 各时段对应的下一次离场时段索引
    price_arr  : (T,)   分时电价

    返回
    ----
    w : (N, T) 意愿度 ∈ [0,1]
    """
    N, T = soc.shape

    # ---- x1：SOC 裕度（相对当前停留段的离场需求）----
    rng_soc = np.maximum(cfg.SOC_MAX - need_at_dep, 0.05)
    x1 = np.clip((soc - need_at_dep) / rng_soc, 0.0, 1.0)

    # ---- x2：停留裕度 = (剩余在站时长 − 补足亏缺所需时长) / 二者之和 ----
    remain_h = np.maximum(t_dep_next - np.arange(T)[None, :], 0.0) * cfg.DT
    deficit_kwh = np.maximum(need_at_dep - soc, 0.0) * e_cap[:, None]
    need_h = deficit_kwh / cfg.P_CH_MAX
    x2 = np.clip((remain_h - need_h) / np.maximum(remain_h + need_h, 1e-6),
                 0.0, 1.0)

    # ---- x3：电价水平（谷=0，峰=1）----
    x3 = np.clip((price_arr[None, :] - cfg.PRICE_VALLEY) /
                 (cfg.PRICE_PEAK - cfg.PRICE_VALLEY), 0.0, 1.0)
    x3 = np.broadcast_to(x3, (N, T))

    # ---- Mamdani 推理（9 条规则，单点输出）----
    l1, m1, h1 = _mfs(x1)
    l2, m2, h2 = _mfs(x2)
    _, _, h3 = _mfs(x3)

    rules = [
        (h1 * h2, 0.90),   # SOC 充裕 & 时间充裕 → 很高
        (h1 * m2, 0.70),
        (m1 * h2, 0.60),
        (m1 * m2, 0.45),
        (h1 * l2, 0.35),   # SOC 充裕但即将离场 → 中低
        (m1 * l2, 0.20),
        (l1 * h2, 0.15),   # SOC 紧张 → 低（保留电量）
        (l1 * m2, 0.10),
        (l1 * l2, 0.05),
    ]
    w = np.zeros((N, T))
    tot = np.zeros((N, T))
    for strength, out in rules:
        w += strength * out
        tot += strength
    w = w / np.maximum(tot, 1e-9)

    # ---- 峰时补偿激励修正：峰时意愿上浮 ----
    w = np.clip(w + 0.15 * (h3 - 0.5), 0.0, 1.0)
    return w


def discharge_power_limit(w, cfg):
    """意愿度 → 允许放电功率上限（kW）：w ≥ 阈值才开放 V2G。"""
    return np.where(w >= cfg.W_DISCHARGE, w * cfg.P_DIS_MAX, 0.0)
