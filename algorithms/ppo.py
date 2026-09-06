# -*- coding: utf-8 -*-
"""策略四：PPO 深度强化学习调度。

MDP 建模（与论文"以策略代理替代集中式优化求解器"思路一致）：
  状态：全局运行点（时刻/电价/基础负荷/接入规模/聚合 SOC/亏缺）
        + 各站局部状态（车辆数/均 SOC/亏缺/充电裕度/放电裕度）
  动作：各站归一化功率指令 ∈ [-1,1]（>0 充电，<0 放电）
  奖励：-(网损 + 电压越限 + 用户成本 - 电站收益) - 离场违约惩罚

算法：clip-PPO（GAE(λ) 优势估计，MLP 策略 128×64，高斯随机策略），
每个回合随机生成新出行场景（域随机化），提升策略泛化能力；
评估阶段在固定场景（CRN）上确定性执行。
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:      # 无 torch 时给出清晰提示，不影响其他模块
    HAS_TORCH = False


if HAS_TORCH:

    class PolicyNet(nn.Module):
        def __init__(self, obs_dim, act_dim, hid=(128, 64)):
            super().__init__()
            layers, last = [], obs_dim
            for h in hid:
                layers += [nn.Linear(last, h), nn.Tanh()]
                last = h
            self.body = nn.Sequential(*layers)
            self.mu = nn.Linear(last, act_dim)
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
            self.vHead = nn.Linear(last, 1)

        def forward(self, x):
            h = self.body(x)
            return self.mu(h), self.log_std.exp().expand_as(self.mu(h)), \
                self.vHead(h).squeeze(-1)

    class PPOAgent:
        def __init__(self, cfg, obs_dim, act_dim, device="cpu"):
            self.cfg = cfg
            self.device = torch.device(device)
            self.net = PolicyNet(obs_dim, act_dim, cfg.PPO_HID).to(self.device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.PPO_LR)

        def act(self, obs, deterministic=False):
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
                mu, std, v = self.net(o)
                if deterministic:
                    a = mu
                else:
                    a = Normal(mu, std).sample()
                lp = Normal(mu, std).log_prob(a).sum(-1)
            return a.squeeze(0).cpu().numpy(), float(lp.item()), float(v.item())

    # ------------------------------------------------------------------
    def _gae(rewards, values, dones, last_v, gamma, lam):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else values[t + 1]
            nonterm = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * nv * nonterm - values[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        return adv

    def train(env_fn, cfg, log_fn=print):
        """训练入口。env_fn: () -> 新环境（每个回合随机场景）。"""
        env = env_fn()
        agent = PPOAgent(cfg, env.obs_dim, env.action_dim)
        total, ep_idx, buf = 0, 0, []
        ep_rewards, ep_viols = [], []
        lr_decayed = False
        while total < cfg.PPO_STEPS_TOTAL:
            # 学习率退火：60% 步数后衰减，抑制后期策略震荡
            if (not lr_decayed) and total > 0.6 * cfg.PPO_STEPS_TOTAL:
                for g in agent.opt.param_groups:
                    g["lr"] *= cfg.PPO_LR_DECAY
                lr_decayed = True
            obs = env.reset()
            done = False
            ep_r, ep_v = 0.0, 0
            while not done:
                a, lp, v = agent.act(obs)
                obs2, r, done, info = env.step(a)
                buf.append((obs, a, lp, v, r, done))
                obs = obs2
                ep_r += r
                ep_v += info["n_viol"]
                total += 1
            ep_idx += 1
            ep_rewards.append(ep_r)
            ep_viols.append(ep_v)

            # ---- 每收集满 PPO_BATCH 步做一次更新（按回合边界截断）----
            if len(buf) >= cfg.PPO_BATCH or total >= cfg.PPO_STEPS_TOTAL:
                obs_s, act_s, lp_s, v_s, r_s, d_s = map(np.array, zip(*buf))
                with torch.no_grad():
                    o = torch.as_tensor(obs_s, dtype=torch.float32)
                    _, _, vv = agent.net(o)
                    v_arr = vv.numpy()
                last_v = 0.0 if d_s[-1] else v_arr[-1]
                adv = _gae(r_s.tolist(), v_arr, d_s, last_v,
                           cfg.PPO_GAMMA, cfg.PPO_LAMBDA)
                ret = adv + v_arr
                # 优势标准化
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                o_t = torch.as_tensor(obs_s, dtype=torch.float32)
                a_t = torch.as_tensor(act_s, dtype=torch.float32)
                lp_t = torch.as_tensor(lp_s, dtype=torch.float32)
                adv_t = torch.as_tensor(adv, dtype=torch.float32)
                ret_t = torch.as_tensor(ret, dtype=torch.float32)

                n = len(o_t)
                idx = np.arange(n)
                rng_shuffle = np.random.default_rng()
                for _ in range(cfg.PPO_EPOCHS):
                    rng_shuffle.shuffle(idx)
                    for st in range(0, n, 256):
                        mb = idx[st:st + 256]
                        mu, std, vv = agent.net(o_t[mb])
                        dist = Normal(mu, std)
                        lp_new = dist.log_prob(a_t[mb]).sum(-1)
                        ratio = (lp_new - lp_t[mb]).exp()
                        s1 = ratio * adv_t[mb]
                        s2 = torch.clamp(ratio, 1 - cfg.PPO_CLIP,
                                         1 + cfg.PPO_CLIP) * adv_t[mb]
                        pi_loss = -torch.min(s1, s2).mean()
                        v_loss = 0.5 * (vv - ret_t[mb]).pow(2).mean()
                        ent = dist.entropy().sum(-1).mean()
                        loss = pi_loss + 0.5 * v_loss - cfg.PPO_ENT_COEF * ent
                        agent.opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.net.parameters(), 0.5)
                        agent.opt.step()
                buf = []
                if ep_idx % 10 == 0:
                    k = min(20, len(ep_rewards))
                    log_fn(f"[PPO] ep={ep_idx} steps={total} "
                           f"reward(20ep mean)={np.mean(ep_rewards[-k:]):.1f} "
                           f"viol(20ep mean)={np.mean(ep_viols[-k:]):.2f}")
        return agent, ep_rewards, ep_viols

    # ------------------------------------------------------------------
    def evaluate(env, agent, seed=None):
        """确定性策略执行一个完整回合，返回站点功率计划。"""
        obs = env.reset()
        done = False
        total_r = 0.0
        while not done:
            a, _, _ = agent.act(obs, deterministic=True)
            obs, r, done, _ = env.step(a)
            total_r += r
        return env.plan_matrix(), total_r

else:   # pragma: no cover - 无 torch 环境占位
    def _no_torch(*a, **k):
        raise RuntimeError("未安装 torch，无法使用 PPO。"
                           "请执行: pip install torch --index-url "
                           "https://download.pytorch.org/whl/cpu")
    train = _no_torch
    evaluate = _no_torch
