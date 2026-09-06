# -*- coding: utf-8 -*-
"""V2G 智能调度平台 · 可视化大屏（Streamlit）。

启动：streamlit run app/dashboard.py
功能：加载 results/ 下预计算的实验结果，交互式展示四种策略的
      负荷曲线、电压轮廓、站点功率、SOC 热力图与关键指标对比。
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")

st.set_page_config(page_title="V2G 智能调度平台", page_icon="⚡", layout="wide")

METHODS = ["Uncoordinated", "TOU Response", "MOPSO", "PPO (ours)"]
CN = {"Uncoordinated": "无序充电", "TOU Response": "分时电价响应",
      "MOPSO": "MOPSO 优化", "PPO (ours)": "PPO 强化学习"}
COLORS = {"Uncoordinated": "#d62728", "TOU Response": "#ff7f0e",
          "MOPSO": "#1f77b4", "PPO (ours)": "#2ca02c"}

st.title("⚡ 车网互动（V2G）智能调度平台")
st.caption("IEEE 33 节点配电网 · 出行链蒙特卡洛场景 · 模糊意愿量化 · "
           "无序充电 / 分时电价 / 改进 MOPSO / PPO 强化学习 对比")

# ---------------- 数据加载 ----------------
@st.cache_data(show_spinner=False)
def load_curves():
    path = os.path.join(RESULTS, "curves.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)


@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(RESULTS, "metrics_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


curves = load_curves()
metrics = load_metrics()

if curves is None or metrics is None:
    st.warning("未找到实验结果，请先运行：`python run_experiment.py`")
    st.stop()

avail_methods = [m for m in METHODS if f"load_{m}" in curves]
th = np.arange(curves["base_load"].shape[0]) * 0.25

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.header("控制台")
    sel = st.multiselect("显示策略", avail_methods,
                         default=avail_methods, format_func=lambda m: CN[m])
    st.divider()
    st.subheader("场景参数")
    st.caption("160 辆 EV / 日，6 座充电站，15 min 步长，"
               "24h 优化视界。参数见 `v2g/config.py`。")
    if st.button("🔄 重新运行实验（较慢）", type="primary"):
        with st.spinner("运行中…"):
            os.system(f'"{sys.executable}" run_experiment.py')
        st.cache_data.clear()
        st.rerun()

# ---------------- KPI 卡片 ----------------
st.subheader("核心指标（多场景均值）")
kpi_show = [("peak_kw", "峰值负荷 (kW)", "{:,.0f}"),
            ("pv_diff_kw", "峰谷差 (kW)", "{:,.0f}"),
            ("loss_kwh", "日网损 (kWh)", "{:,.0f}"),
            ("v2g_energy_kwh", "V2G 电量 (kWh)", "{:,.0f}"),
            ("user_cost_mean", "户均成本 (元)", "{:.1f}"),
            ("station_rev", "电站收益 (元)", "{:,.0f}")]
cols = st.columns(len(kpi_show))
for c, (k, label, fmt) in zip(cols, kpi_show):
    sub = metrics[metrics.metric == k].set_index("method")["mean"]
    best = None
    if k in ("peak_kw", "pv_diff_kw", "loss_kwh", "user_cost_mean"):
        best = sub.idxmin() if len(sub) else None
    elif k in ("v2g_energy_kwh", "station_rev"):
        best = sub.idxmax() if len(sub) else None
    for m in sel:
        if m in sub.index:
            with c:
                star = " 🏆" if m == best else ""
                st.metric(f"{label} · {CN[m]}", fmt.format(sub[m]) + star)

st.divider()

# ---------------- 负荷曲线 ----------------
st.subheader("系统负荷曲线")
fig, ax = plt.subplots(figsize=(9, 3.4), dpi=120)
ax.plot(th, curves["base_load"], "k--", lw=1.2, label="基础负荷")
for m in sel:
    ax.plot(th, curves[f"load_{m}"], color=COLORS[m], lw=1.7, label=CN[m])
ax.set_xlabel("时刻 (h)"); ax.set_ylabel("总负荷 (kW)")
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2)); ax.legend(fontsize=8)
ax.grid(alpha=0.3)
st.pyplot(fig)

# ---------------- 电压 + 站点功率 ----------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("馈线最低电压")
    fig, ax = plt.subplots(figsize=(4.6, 3), dpi=120)
    ax.axhline(0.95, color="gray", ls=":", lw=1)
    for m in sel:
        ax.plot(th, curves[f"vmin_{m}"], color=COLORS[m], lw=1.4, label=CN[m])
    ax.set_xlabel("时刻 (h)"); ax.set_ylabel("最低电压 (p.u.)")
    ax.set_xlim(0, 24); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    st.pyplot(fig)
with c2:
    st.subheader("站点级功率（PPO）")
    fig, ax = plt.subplots(figsize=(4.6, 3), dpi=120)
    key = f"pst_PPO (ours)"
    if key in curves and "PPO (ours)" in sel:
        pst = curves[key]
        ax.stackplot(th, np.where(pst > 0, pst, 0),
                     colors=plt.cm.Blues(np.linspace(0.35, 0.75, len(pst))))
        ax.plot(th, -np.where(pst < 0, -pst, 0).sum(axis=0),
                color="#d62728", lw=1.4, label="V2G 放电")
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(fontsize=7)
    ax.set_xlabel("时刻 (h)"); ax.set_ylabel("功率 (kW)"); ax.grid(alpha=0.3)
    st.pyplot(fig)

# ---------------- SOC 热力图 + 指标表 ----------------
c3, c4 = st.columns(2)
with c3:
    st.subheader("SOC 轨迹热力图（PPO，前 40 辆）")
    key = "soc_PPO (ours)"
    if key in curves:
        fig, ax = plt.subplots(figsize=(4.6, 3), dpi=120)
        im = ax.imshow(curves[key][:40], aspect="auto", origin="lower",
                       cmap="viridis", extent=[0, 24, 0, 40], vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="SOC")
        ax.set_xlabel("时刻 (h)"); ax.set_ylabel("EV 编号")
        st.pyplot(fig)
with c4:
    st.subheader("指标明细")
    piv = metrics.pivot_table(index="metric", columns="method", values="mean")
    piv.columns = [CN.get(c, c) for c in piv.columns]
    st.dataframe(piv.round(2), height=380)

st.divider()
st.caption("研究内容：考虑用户出行概率与参与意愿的 V2G 充放电协同调度 | "
           "线性化 DistFlow 潮流 | 改进多目标粒子群 | 深度强化学习 (PPO) | "
           "开源代码见仓库 README")
