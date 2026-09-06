# -*- coding: utf-8 -*-
"""IEEE 33 节点配电系统 + 线性化 DistFlow 潮流。

电力模型采用 Baran & Wu (1989) 简化支路潮流（Linearized DistFlow）：
    V_j ≈ V_i - (r_k P_k + x_k Q_k) / V_0
    P_loss = Σ_k r_k (P_k² + Q_k²) / V_0²
其中 P_k、Q_k 为支路 k 的下游有功/无功注入之和，V_0 取 1.0 pu。

该模型完全向量化（矩阵运算），单次求解 <1 ms，可嵌入 PSO/RL 循环内；
同时保留 pandapower 精确交流潮流接口用于结果校验（可选依赖）。
"""

import os
import numpy as np

# 支路数据 (from, to, R/ohm, X/ohm) —— MATPOWER case33bw（Baran & Wu 1989），1-based
# 拓扑：主馈线 1-2-...-18；支线A 2-19-...-22；支线B 3-23-24-25；支线C 6-26-...-33
_BRANCHES = [
    (1, 2, 0.0922, 0.0470), (2, 3, 0.4930, 0.2511), (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941), (5, 6, 0.8190, 0.7070), (6, 7, 0.1872, 0.6188),
    (7, 8, 0.7114, 0.2351), (8, 9, 1.0300, 0.7400), (9, 10, 1.0440, 0.7400),
    (10, 11, 0.1966, 0.0650), (11, 12, 0.3744, 0.1238), (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.5910, 0.5260), (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210), (17, 18, 0.7320, 0.5740), (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784), (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083), (23, 24, 0.8980, 0.7091), (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 0.7006), (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619), (32, 33, 0.3410, 0.5302),
]

# 节点负荷 (bus 1-based, P/kW, Q/kVAr)，总线负荷 3715 kW + j2300 kVAr（含节点33）
_LOADS = [
    (2, 100, 60), (3, 90, 40), (4, 120, 80), (5, 60, 30), (6, 60, 20),
    (7, 200, 100), (8, 200, 100), (9, 60, 20), (10, 60, 20), (11, 45, 30),
    (12, 60, 35), (13, 60, 35), (14, 120, 80), (15, 60, 10), (16, 60, 20),
    (17, 60, 20), (18, 90, 40), (19, 90, 40), (20, 90, 40), (21, 90, 40),
    (22, 90, 40), (23, 90, 50), (24, 420, 200), (25, 420, 200), (26, 60, 25),
    (27, 60, 25), (28, 60, 20), (29, 120, 70), (30, 200, 600), (31, 150, 70),
    (32, 210, 100), (33, 60, 40),
]

# 典型居民/商业混合日负荷曲线（24 点，峰值标幺 1.0）
_DAY_PROFILE = np.array([
    0.53, 0.49, 0.47, 0.46, 0.46, 0.48, 0.55, 0.62, 0.66, 0.64,
    0.62, 0.60, 0.58, 0.57, 0.56, 0.58, 0.64, 0.72, 0.86, 1.00,
    0.96, 0.86, 0.74, 0.62,
])


class Grid33:
    """IEEE 33 节点系统：拓扑、基础负荷与线性化潮流。"""

    NB = 33          # 节点数（0-based：0 为平衡节点）
    NL = 32          # 支路数
    S_BASE = 10.0    # 功率基准 MVA
    V_LL = 12.66     # 线电压 kV
    _Z_BASE = V_LL ** 2 / S_BASE   # 阻抗基准 ≈ 16.0276 ohm

    def __init__(self, cfg):
        self.cfg = cfg
        T = cfg.T
        self.T = T

        # ---- 拓扑与阻抗（欧姆）----
        bf = np.array([b[0] for b in _BRANCHES]) - 1   # from（0-based）
        bt = np.array([b[1] for b in _BRANCHES]) - 1   # to
        br = np.array([b[2] for b in _BRANCHES])
        bx = np.array([b[3] for b in _BRANCHES])
        self.br_from, self.br_to, self.br_r, self.br_x = bf, bt, br, bx

        # ---- 每个节点的父支路（径向网络唯一）----
        self.parent_branch = np.full(self.NB, -1, dtype=int)
        for k, (_, j) in enumerate(zip(bf, bt)):
            self.parent_branch[j] = k

        # ---- 每个节点的下游支路集合（含自身方向）----
        # children[k] = 该支路 to 节点的子支路列表
        self.children = [[] for _ in range(self.NL)]
        for k, i in enumerate(bt):
            for m, j in enumerate(bf):
                if j == i and m != k:
                    self.children[k].append(m)

        # ---- 节点-支路注入映射矩阵 A（NL×NB）：A[k,i]=1 表示节点 i 在支路 k 下游 ----
        # 通过从叶节点向根逐级累加构造
        self.A = np.zeros((self.NL, self.NB))
        self.A[np.arange(self.NL), bt] = 1.0
        # 拓扑排序：反复把子支路系数加到父支路
        pending = set(range(self.NL))
        while pending:
            progressed = False
            for k in list(pending):
                downstream_done = all(m not in pending for m in self.children[k])
                if downstream_done:
                    for m in self.children[k]:
                        self.A[k] += self.A[m]
                    pending.discard(k)
                    progressed = True
            if not progressed:   # 防御：不应发生（径向连通网络）
                break

        # ---- 基础负荷曲线（T 时段，kW / kVAr）----
        mult = np.interp(
            np.linspace(0, 24, T, endpoint=False) + cfg.DT / 2,
            np.arange(24) + 0.5, _DAY_PROFILE,
        )
        self.base_P = np.zeros((self.NB, T))   # kW
        self.base_Q = np.zeros((self.NB, T))   # kVAr
        for bus, p, q in _LOADS:
            self.base_P[bus - 1] = p * mult
            self.base_Q[bus - 1] = q * mult

    # ------------------------------------------------------------------
    def powerflow(self, ev_p_kw):
        """线性化 DistFlow 潮流。

        参数
        ----
        ev_p_kw : (NB, T) 数组，EV 净注入功率（充电为正、放电为负），kW

        返回
        ----
        volt : (NB, T) 节点电压幅值 pu（平衡节点为 1.0）
        loss : (T,)    系统有功网损 kW
        """
        T = self.T
        P_node = self.base_P + ev_p_kw * 1.0                     # kW
        Q_node = self.base_Q + ev_p_kw * 0.05                    # EV 换流器吸收少量无功（cosφ≈0.999）
        # 支路潮流：S_k = Σ_{下游节点} S_node
        P_br = self.A @ P_node / 1000.0                          # MW
        Q_br = self.A @ Q_node / 1000.0                          # MVAr
        # 线性化电压降落：ΔV = Σ (r_pu·P_pu + x_pu·Q_pu)
        r_pu = self.br_r / self._Z_BASE                          # ohm -> pu
        x_pu = self.br_x / self._Z_BASE
        dV = (r_pu[:, None] * P_br + x_pu[:, None] * Q_br) / self.S_BASE   # (NL, T)
        # 节点电压 = 1 - 沿路径所有支路降落之和（根->叶累加）
        volt = np.ones((self.NB, T))
        drop = np.zeros((self.NL, T))
        for k in self._topo_order():
            parent_k = self.parent_branch[self.br_from[k]]
            drop[k] = dV[k] + (drop[parent_k] if parent_k >= 0 else 0.0)
            volt[self.br_to[k]] = 1.0 - drop[k]
        # 网损：Σ r·(P²+Q²)/V_LL²，三相功率 kW = (P_MW²+Q²)·R_ohm/V_kV²·1e3，V 取额定近似
        loss = np.sum(self.br_r[:, None] * (P_br ** 2 + Q_br ** 2) / 12.66 ** 2 * 1e3, axis=0)
        return volt, loss

    def _topo_order(self):
        """支路的根->叶遍历顺序。"""
        if hasattr(self, "_order_cache"):
            return self._order_cache
        order = []
        visited = [False] * self.NL
        stack = [k for k in range(self.NL) if self.br_from[k] == 0]
        while stack:
            k = stack.pop(0)
            if visited[k]:
                continue
            visited[k] = True
            order.append(k)
            stack.extend(self.children[k])
        self._order_cache = order
        return order

    # ------------------------------------------------------------------
    def verify_pandapower(self, ev_p_kw, t_list=(48, 76)):
        """用 pandapower 精确交流潮流校验指定时刻的最低电压（可选）。

        pandapower 未安装或其依赖不可用时返回 None，不影响主流程。
        """
        try:
            import pandapower.networks as ppn
            import pandapower as pp
        except Exception:            # ImportError 或 numba 等依赖崩溃
            return None
        net = ppn.case33bw()
        out = {}
        for t in t_list:
            net.load.p_mw[:] = 0.0
            net.load.q_mvar[:] = 0.0
            for bus, p, q in _LOADS:
                idx = net.load.index[net.load.bus == bus - 1]
                net.load.p_mw[idx] = (p * 1.0 + ev_p_kw[bus - 1, t]) / 1000.0
                net.load.q_mvar[idx] = q / 1000.0
            pp.runpp(net, calculate_voltage_angles=False)
            out[t] = float(net.res_bus.vm_pu.min())
        return out
