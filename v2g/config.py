# -*- coding: utf-8 -*-
"""全局参数配置。

所有模型参数集中在此，便于面试时说明"参数来源与灵敏度分析"。
时间尺度：24h，步长 15min，T = 96。
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ---------------- 仿真时间 ----------------
    T: int = 96                    # 一天 96 个时段（15 min）
    DT: float = 0.25               # 每时段时长（小时）
    DAYS: int = 1                  # 仿真天数

    # ---------------- 电动汽车群 ----------------
    N_EV: int = 160                # 每日参与仿真的 EV 总数
    E_CAP_MEAN: float = 55.0       # 电池容量均值 kWh（对数正态）
    E_CAP_SIGMA: float = 0.25
    P_CH_MAX: float = 7.0          # 最大充电功率 kW（交流桩）
    P_DIS_MAX: float = 7.0         # 最大放电功率 kW（V2G）
    ETA_CH: float = 0.95           # 充电效率
    ETA_DIS: float = 0.95          # 放电效率
    SOC_MIN: float = 0.10          # 电池下限
    SOC_MAX: float = 0.95          # 电池上限（V2G 保护）
    SOC_DEPART_MEAN: float = 0.75  # 出发时刻期望 SOC 均值
    SOC_DEPART_SIG: float = 0.12
    KM_PER_DAY_MEAN: float = 42.0  # 日行驶里程均值 km（对数正态）
    KM_PER_DAY_SIGMA: float = 0.55
    CONSUMPTION: float = 0.15      # kWh/km
    N_SCENARIO: int = 5            # 评估场景数（公共随机数法保证公平对比）

    # ---------------- 充电站（IEEE33 节点，0-based）----------------
    # 覆盖三条馈线：主馈线(7,12,17)、支线A(20)、支线B(24)、支线C(30)，1-based
    STATION_BUS: tuple = (6, 11, 16, 19, 23, 29)
    STATION_RATIO: tuple = (0.2, 0.2, 0.15, 0.15, 0.15, 0.15)  # EV 分配比例

    # ---------------- 分时电价（元/kWh）----------------
    PRICE_VALLEY: float = 0.31
    PRICE_FLAT: float = 0.62
    PRICE_PEAK: float = 1.05
    VALLEY_HOURS: tuple = ((23, 24), (0, 7))
    FLAT_HOURS: tuple = ((7, 10), (15, 18))
    PEAK_HOURS: tuple = ((10, 15), (18, 23))
    V2G_COMPENSATION: float = 0.70     # V2G 放电补偿电价 元/kWh
    CHARGE_SERVICE_RATE: float = 1.40  # 用户侧充电价 = TOU × 该系数（含服务费）
    GRID_BUY_RATE: float = 0.90        # 电站向电网售电（放电）结算系数 × TOU
    W_DISCHARGE: float = 0.50          # 允许放电的最低意愿度阈值

    # ---------------- 成本系数 ----------------
    DEGRADATION_COST: float = 0.12     # 电池退化成本 元/kWh 吞吐
    LOSS_COST: float = 0.62            # 网损电价 元/kWh
    W_NETWORK: float = 10.0            # 目标1 权重：网损+波动
    W_USER: float = 1.0                # 目标2 权重：用户成本
    W_STATION: float = 1.0             # 目标3 权重：电站收益
    V_LIMIT: float = 0.05              # 电压偏差限值 0.95~1.05 pu

    # ---------------- MOPSO ----------------
    PSO_POP: int = 30
    PSO_ITER: int = 60
    PSO_ARCHIVE: int = 50
    PSO_W: float = 0.7                 # 惯性权重（线性递减 0.9->0.4 由代码控制）
    PSO_C1: float = 1.5
    PSO_C2: float = 1.5
    PSO_TURB: float = 0.1              # 变异（扰动）概率：改进项1
    PSO_KNEE: str = "fuzzy"            # Pareto 解选择方法: fuzzy / knee

    # ---------------- PPO ----------------
    PPO_HID: tuple = (128, 64)
    PPO_LR: float = 2e-4
    PPO_LR_DECAY: float = 0.3        # 训练后期学习率衰减系数
    PPO_SHAPE_K: float = 0.02        # 势函数整形系数（亏缺能量 kWh）
    PPO_GAMMA: float = 0.99
    PPO_LAMBDA: float = 0.95
    PPO_CLIP: float = 0.2
    PPO_EPOCHS: int = 8
    PPO_BATCH: int = 2048
    PPO_STEPS_TOTAL: int = 350_000     # 总训练步数
    PPO_ENT_COEF: float = 0.002
    PPO_SEED: int = 7

    # ---------------- 其他 ----------------
    SEED: int = 42

    def price_array(self):
        """返回长度 T 的分时电价数组（元/kWh）。"""
        import numpy as np
        p = np.zeros(self.T)
        for t in range(self.T):
            h = t * self.DT
            if any(a <= h < b for a, b in self.VALLEY_HOURS):
                p[t] = self.PRICE_VALLEY
            elif any(a <= h < b for a, b in self.PEAK_HOURS):
                p[t] = self.PRICE_PEAK
            else:
                p[t] = self.PRICE_FLAT
        return p
