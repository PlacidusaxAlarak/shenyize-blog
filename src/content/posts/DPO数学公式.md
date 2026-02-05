---
title: PPO (RLHF) 到 DPO 的完整数学推导
published: 2026-02-03
description: '严格按照 DPO 论文推导 Section 3-4 及 Appendix A.1-A.2 的核心公式。'
tags: [LLM, Reinforcement Learning, DPO, Math]
category: 'Reinforcement Learning'
draft: false
lang: 'zh'
priority: 5
---

# PPO (RLHF) 到 DPO 的完整数学推导

本文档严格按照 DPO 论文（*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*）的逻辑，推导 Section 3-4 及 Appendix A.1-A.2 的核心公式。

推导路径：
1.  **PPO 目标**：写出 RLHF 阶段带有 KL 惩罚的期望奖励最大化目标。
2.  **闭式最优解**：证明该目标在非参数化空间下的最优策略 $\pi^*$ 的解析形式。
3.  **反解 Reward**：将 Reward 函数用最优策略 $\pi^*$ 和参考策略 $\pi_{\text{ref}}$ 表示。
4.  **代入偏好模型**：将 Reward 表达式代入 Bradley-Terry 模型，消去配分函数 $Z(x)$。
5.  **DPO Loss**：通过最大似然估计得到最终的 DPO 损失函数。

---

## 0. 记号与背景

* $x$：Prompt（输入上下文）
* $y$：Completion（模型生成的回答）
* $\pi(y|x)$：我们要学习的策略模型
* $\pi_{\text{ref}}(y|x)$：参考策略（通常是 SFT 模型，保持冻结）
* $r(x,y)$：奖励函数（在 RLHF 中是 Reward Model，在 DPO 推导中假设存在这样一个隐式的真实奖励）
* $\beta$：KL 散度约束系数（控制策略不偏离 $\pi_{\text{ref}}$ 的程度）
* $\sigma(t) = \frac{1}{1+e^{-t}}$：Sigmoid 函数

---

## 1. PPO 在 RLHF 中的优化目标

在 RLHF 的 RL 阶段（PPO），我们的目标是最大化期望奖励，同时约束模型与参考模型的 KL 散度。论文 Eq.(3) 将此目标形式化为：

$$
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}} \left[ 
\mathbb{E}_{y \sim \pi(\cdot|x)} [r(x,y)] - \beta D_{\mathrm{KL}}(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)) 
\right] \tag{1}
$$

### 2.1 展开 KL 散度
对于固定的 $x$，KL 散度定义为：
$$
D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}}) = \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}
$$

将 (1) 中的期望展开为求和形式：
$$
\max_{\pi} \mathbb{E}_{x} \left[ 
\sum_y \pi(y|x) r(x,y) - \beta \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} 
\right]
$$

合并同类项 $\pi(y|x)$：
$$
\max_{\pi} \mathbb{E}_{x} \left[ 
\sum_y \pi(y|x) \left( r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right) 
\right] \tag{2}
$$

---

## 2. 推导闭式最优策略 (Deriving the Optimal Policy)

为了得到最优解 $\pi^*$，我们将“最大化问题”转换为“最小化 KL 问题”。

### 2.2 目标变换
利用性质 $\arg\max F(\pi) = \arg\min [-F(\pi)]$，我们将 (2) 取负号并除以常数 $\beta$：

$$
\min_{\pi} \mathbb{E}_{x} \left[ 
\sum_y \pi(y|x) \left( \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta} r(x,y) \right) 
\right] \tag{3}
$$

### 2.3 引入配分函数 (Partition Function)
定义 $Z(x)$ 为归一化常数（论文 Eq.(4) 下方定义）：
$$
Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right) \tag{4}
$$

定义最优策略的形式 $\pi^*(y|x)$（我们稍后证明它就是最优解）：
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right) \tag{5}
$$

### 2.4 构造 KL 形式
对 $\pi^*(y|x)$ 取对数：
$$
\log \pi^*(y|x) = -\log Z(x) + \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x,y)
$$

先移项解出 $r(x,y)$ 的表达式：
$$
\frac{1}{\beta} r(x,y) = \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) + \log Z(x)
$$

将其逐步代入 $\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta} r(x,y)$：
$$
\begin{aligned}
\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta} r(x,y)
&= \log \pi(y|x) - \log \pi_{\text{ref}}(y|x) - \frac{1}{\beta} r(x,y) \\
&= \log \pi(y|x) - \log \pi_{\text{ref}}(y|x) - \Big(\log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) + \log Z(x)\Big) \\
&= \log \pi(y|x) - \log \pi^*(y|x) - \log Z(x) \\
&= \log \frac{\pi(y|x)}{\pi^*(y|x)} - \log Z(x)
\end{aligned}
$$
即得到关键恒等式：
$$
\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta} r(x,y) = \log \frac{\pi(y|x)}{\pi^*(y|x)} - \log Z(x) \tag{6}
$$

将 (6) 代回目标函数 (3)：
$$
\min_{\pi} \mathbb{E}_{x} \left[ 
\sum_y \pi(y|x) \left( \log \frac{\pi(y|x)}{\pi^*(y|x)} - \log Z(x) \right) 
\right]
$$

逐项展开求和：
$$
\sum_y \pi(y|x)\left(\log \frac{\pi(y|x)}{\pi^*(y|x)} - \log Z(x)\right)
= \sum_y \pi(y|x)\log \frac{\pi(y|x)}{\pi^*(y|x)} - \log Z(x)\sum_y \pi(y|x).
$$
由于 $\sum_y \pi(y|x)=1$，上式化为
$$
\sum_y \pi(y|x)\log \frac{\pi(y|x)}{\pi^*(y|x)} - \log Z(x)
= D_{\mathrm{KL}}(\pi(\cdot|x)\|\pi^*(\cdot|x)) - \log Z(x).
$$
由于 $\sum_y \pi(y|x) \log Z(x) = \log Z(x)$（与 $y$ 无关），目标变为：
$$
\min_{\pi} \mathbb{E}_{x} \left[ 
D_{\mathrm{KL}}(\pi(\cdot|x) \| \pi^*(\cdot|x)) - \log Z(x) 
\right] \tag{7}
$$

:::note
提示：虽然 $Z(x)$ 的定义里含有 $r(x,y)$，但其中的 $y$ 是内部求和变量，求和完成后只剩下 $x$ 的函数。因此 $\log Z(x)$ 对外层 $\sum_y \pi(y|x)$ 来说是常数，有 $\sum_y \pi(y|x)\log Z(x)=\log Z(x)$。
:::

### 2.5.1 两条等价路线
:::note
路线 A（拉格朗日乘子法，逐步求解）：
对固定的 $x$，最小化式 (3) 的内层目标
$$
\min_{\pi(\cdot|x)} \sum_y \pi(y|x)\left(\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}-\frac{1}{\beta}r(x,y)\right),
$$
并满足约束 $\sum_y \pi(y|x)=1$。将目标展开为显式形式：
$$
\sum_y \pi(y|x)\log \pi(y|x)-\sum_y \pi(y|x)\log \pi_{\text{ref}}(y|x)-\frac{1}{\beta}\sum_y \pi(y|x)r(x,y).
$$
构造拉格朗日函数：
$$
\mathcal{L}=\sum_y \pi(y|x)\log \pi(y|x)-\sum_y \pi(y|x)\log \pi_{\text{ref}}(y|x)-\frac{1}{\beta}\sum_y \pi(y|x)r(x,y)+\lambda\Big(\sum_y \pi(y|x)-1\Big).
$$
对每个 $y$ 求偏导并令其为 0（注意 $\frac{\partial}{\partial \pi}\pi\log\pi=\log\pi+1$）：
$$
\frac{\partial \mathcal{L}}{\partial \pi(y|x)}=\log \pi(y|x)+1-\log \pi_{\text{ref}}(y|x)-\frac{1}{\beta}r(x,y)+\lambda=0.
$$
移项得到
$$
\log \pi(y|x)=\log \pi_{\text{ref}}(y|x)+\frac{1}{\beta}r(x,y)-(1+\lambda).
$$
两边取指数：
$$
\pi(y|x)=\exp(-(1+\lambda))\,\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right).
$$
令 $Z(x)=\exp(1+\lambda)$，得到
$$
\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right).
$$
最后用归一化条件 $\sum_y \pi^*(y|x)=1$ 解出
$$
Z(x)=\sum_y \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right).
$$
:::

:::note
路线 B（改写为 KL + 常数，逐步等价变形）：
从定义 (5) 出发，先写出
$$
\log \pi^*(y|x)=-\log Z(x)+\log \pi_{\text{ref}}(y|x)+\frac{1}{\beta}r(x,y).
$$
移项得到
$$
\frac{1}{\beta}r(x,y)=\log \pi^*(y|x)-\log \pi_{\text{ref}}(y|x)+\log Z(x).
$$
代入到式 (3) 的被积项：
$$
\begin{aligned}
\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}-\frac{1}{\beta}r(x,y)
&=\log \pi(y|x)-\log \pi_{\text{ref}}(y|x)-\frac{1}{\beta}r(x,y) \\
&=\log \pi(y|x)-\log \pi_{\text{ref}}(y|x)-\Big(\log \pi^*(y|x)-\log \pi_{\text{ref}}(y|x)+\log Z(x)\Big) \\
&=\log \pi(y|x)-\log \pi^*(y|x)-\log Z(x) \\
&=\log \frac{\pi(y|x)}{\pi^*(y|x)}-\log Z(x).
\end{aligned}
$$
对 $y$ 求和：
$$
\sum_y \pi(y|x)\left(\log \frac{\pi(y|x)}{\pi^*(y|x)}-\log Z(x)\right)
=\sum_y \pi(y|x)\log \frac{\pi(y|x)}{\pi^*(y|x)}-\log Z(x)\sum_y \pi(y|x),
$$
而 $\sum_y \pi(y|x)=1$，因此目标变为
$$
D_{\mathrm{KL}}(\pi(\cdot|x)\|\pi^*(\cdot|x))-\log Z(x).
$$
利用 KL 非负性，最优解满足
$$
D_{\mathrm{KL}}(\pi(\cdot|x)\|\pi^*(\cdot|x))=0 \Rightarrow \pi=\pi^*.
$$
:::

### 2.5 利用 Gibbs 不等式求解
:::note
Gibbs 不等式（KL 非负性）证明：对离散分布 $p,q$，记 $u=\frac{q_i}{p_i}$，利用不等式 $\log u \le u-1$（当且仅当 $u=1$ 取等），有
$$
\log \frac{p_i}{q_i} = -\log \frac{q_i}{p_i} \ge 1-\frac{q_i}{p_i}.
$$
两边乘以 $p_i$ 并对 $i$ 求和：
$$
\sum_i p_i \log \frac{p_i}{q_i} \ge \sum_i (p_i-q_i)=0.
$$
因此 $D_{\mathrm{KL}}(p\|q)\ge 0$，且当且仅当 $p=q$ 取等。
:::

根据 Gibbs 不等式（即 KL 散度非负性）：
$$
D_{\mathrm{KL}}(p \| q) \ge 0, \quad D_{\mathrm{KL}}(p \| q) = 0 \iff p = q
$$
要使 (7) 最小，只需让 KL 项为 0。因此，最优策略即为我们在 (5) 中定义的：

$$
\boxed{\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right)} \tag{论文 Eq.4}
$$

:::note
关于“为什么 $\pi^*=\pi$”：由式 (7) 得目标为
$$
\min_{\pi}\; D_{\mathrm{KL}}(\pi(\cdot|x)\|\pi^*(\cdot|x))-\log Z(x).
$$
其中 $-\log Z(x)$ 与 $\pi$ 无关，因此等价于最小化 KL 散度。由 Gibbs 不等式，
$$
D_{\mathrm{KL}}(\pi\|\pi^*)\ge 0,\quad D_{\mathrm{KL}}(\pi\|\pi^*)=0 \iff \pi=\pi^*,
$$
故最优解满足 $\pi=\pi^*$。式 (4) 只是给出了这个最优分布 $\pi^*$ 的显式形式。
:::

---

## 3. 从最优策略反解 Reward

DPO 的核心思想是用策略来表示奖励。从 Eq.(4) 出发，两边取对数并移项：

$$
\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x,y) - \log Z(x)
$$

解出 $r(x,y)$：
$$
\frac{1}{\beta} r(x,y) = \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x)
$$

$$
\boxed{r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)} \tag{论文 Eq.5}
$$

---

## 4. 代入 Bradley-Terry 偏好模型

### 4.1 偏好模型定义
Bradley-Terry (BT) 模型假设人类选择回答 $y_1$ 优于 $y_2$ 的概率为：
$$
p(y_1 \succ y_2 | x) = \frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))} = \sigma(r(x, y_1) - r(x, y_2)) \tag{8}
$$

### 4.2 配分函数 $Z(x)$ 的抵消
计算两个回答的奖励差值。利用 Eq.(5) 的反解公式：

$$
\begin{aligned}
r(x, y_1) - r(x, y_2) &= \left( \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} + \beta \log Z(x) \right) \\
&= \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)}
\end{aligned}
$$

**关键点**：$Z(x)$ 仅依赖于 $x$，在差分中直接抵消了。这使得我们不需要估计复杂的配分函数。

将此差值代入 BT 模型 (8)：
$$
\boxed{p(y_1 \succ y_2 | x) = \sigma \left( \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} \right)} \tag{论文 Eq.6}
$$

---

## 5. DPO 损失函数 (Final Objective)

现在我们将最优策略 $\pi^*$ 替换为我们需要训练的参数化策略 $\pi_\theta$。
给定偏好数据集 $\mathcal{D} = \{(x, y_w, y_l)\}$，其中 $y_w$ 是胜者，$y_l$ 是败者。

最大化偏好数据的对数似然（Log-Likelihood）：
$$
\max_{\theta} \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log p_\theta(y_w \succ y_l | x) \right]
$$

等价于最小化负对数似然（Negative Log-Likelihood）。代入 Eq.(6) 的形式：

$$
L_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

这就是最终的 DPO 损失函数（论文 Eq.7）。

$$
\boxed{L_{\text{DPO}} = -\mathbb{E}_{\mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]}
$$
