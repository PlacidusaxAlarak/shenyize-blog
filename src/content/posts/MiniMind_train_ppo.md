---
title: MiniMind PPO 训练源码深度解析
published: 2026-01-29
description: '深入剖析 MiniMind 项目的 PPO (Proximal Policy Optimization) 算法实现。'
image: ''
tags: [LLM, Reinforcement Learning]
category: 'Reinforcement Learning'
draft: false
lang: 'zh'
priority: 5
---
![PPO算法完整流程图](./images/PPO.png)
*图 1：PPO 算法核心流程架构图（包含数据收集、GAE 计算及 Actor-Critic 双流更新）*

## PPO 算法与 MiniMind 对齐实战导读

在大型语言模型（LLM）的训练流水线中，如果说预训练（Pre-training）赋予了模型广博的知识，SFT（Supervised Fine-Tuning）教会了模型遵循指令，那么 **RLHF（Reinforcement Learning from Human Feedback）** 则是让模型真正“对齐”人类价值观、学会权衡优劣的关键步骤。

而在 RLHF 的众多算法中，**PPO (Proximal Policy Optimization)** 凭借其在稳定性与样本效率之间的出色平衡，成为了目前事实上的行业标准。

本文将深入剖析 MiniMind 项目中的 `train_ppo.py` 源码。不同于教科书式的理论讲解，我们将直接从代码实现的角度，解构一个支持 **推理能力增强（Reasoning-aware）** 的 PPO 训练系统。

通过阅读本文，你将理解 MiniMind 是如何通过以下核心模块构建其强化学习闭环的：

1. **完整的 Actor-Critic 架构**：
代码中不仅实例化了用于生成的 **Actor Model**（策略网络），还构建了一个基于 `MiniMindLM` 但修改了输出层（`value_head`）的 **Critic Model**（价值网络）。为了保证训练的数学严谨性，系统还维护了 **Old Actor**（用于计算概率比率 ）和 **Reference Model**（用于计算 KL 散度惩罚），构成了经典的“四模型”交互结构。
2. **混合奖励工程 (Hybrid Reward Engineering)**：
这是本实现的亮点之一。在 `calculate_rewards` 函数中，我们不仅引入了外部的 **Reward Model** 对回复质量打分，还针对推理模型（Reasoning Model）设计了 **基于规则的格式奖励**。通过正则表达式（Regex）严格约束 `<think>` 和 `<answer>` 标签的结构，并引入稀疏标记奖励，强制模型在强化学习过程中学会“先思考，后回答”的思维链模式。
3. **PPO 核心目标函数**：
我们将逐行拆解 `ppo_train_epoch` 中的损失计算逻辑：
* **Policy Loss**：利用这一时刻的优势函数（Advantage）和概率比率（Ratio），配合 PPO 标志性的 **Clipping 机制**（`clip_epsilon`），限制策略更新幅度，防止模型“学崩”。
* **Value Loss**：通过 MSE 损失让 Critic 网络更准确地预估当前状态的价值。
* **KL Divergence Penalty**：为了防止模型在优化过程中过度偏离 SFT 后的基座模型（Reward Hacking），我们在总损失中加入了动态的 KL 散度惩罚项。


4. **工程化实现细节**：
从 `DistributedDataParallel` (DDP) 的分布式封装，到处理变长序列的 `Mask` 技巧，再到左侧填充（Left Padding）对生成过程的影响，本文将展示如何在一个真实的 PyTorch 环境中高效、稳定地运行 PPO。

让我们跟随代码，看一看这套复杂的“数字齿轮”是如何精密咬合的。

---

## 全局引用与环境初始化 (Imports & Setup)

<details>
<summary><strong>👉 点击展开查看完整引用与环境初始化代码</strong></summary>

```python title="train_ppo.py"
import os
import sys

__package__ = "trainer" #当前代码属于trainer这个包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
#项目根目录

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')
```
</details>

## 核心架构设计 (The Critic Model)

```python title="train_ppo.py"
# 自定义的Critic模型，继承自MiniMindForCausalLM
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 将原有的lm_head(输出vocab_size维度)替换为value_head(输出1维度，即对当前状态进行价值打分)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态:[Batch_size, Sequence_Length, Hidden_size]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        # value_head之后，张量的形状变为[B, L, 1], squeeze(-1)去掉最后一个维度，即values变为[B, L]
        values = self.value_head(hidden_states).squeeze(-1)
        return values

```


## 混合奖励机制 (Hybrid Reward Engineering)

为了让模型习得“DeepSeek-R1”式的思考模式，单纯依赖传统的奖励模型（Reward Model）是不够的。我们需要通过混合奖励机制，显式地引导模型生成符合 `<think>...</think><answer>...</answer>` 结构的回复。

### 1. 格式规范奖励 (Format Compliance Reward)

首先，我们定义了一个内部函数 `reasoning_model_reward`，它利用正则表达式（Regex）来强制约束模型的输出格式。如果模型能完美生成包含思考和回答标签的结构，给予 `0.5` 的硬性奖励。这有助于模型在初期快速通过强化学习“学会”这种特定的输出范式。

```python
    def reasoning_model_reward(rewards):
        # 1. 格式奖励（仅针对训练推理模型时使用）
        # 两种匹配模式, pattern是</think>与<answer>之间没有多余空行，pattern2是允许一个多余空行
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        #列表只会出现两种情况，re.Match对象或者None(匹配失败)
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5) # 允许中间多一个换行
            else:
                format_rewards.append(0.0)
        #转化为张量
        rewards += torch.tensor(format_rewards, device=args.device)

```

### 2. 稀疏标记奖励 (Sparse Tag Reward)

为了防止训练初期的奖励过于稀疏（即模型很难一开始就完美匹配整个正则），我们引入了细粒度的标记奖励。只要模型输出了正确的 `<think>` 或 `<answer>` 标签，每个标签单独给予 `0.25` 的奖励。这种“积少成多”的策略能有效引导模型逐步逼近最终的正确格式。

```python
        # 2. 标记奖励（防止严格奖励稀疏，仅针对训练推理模型时使用）
        def mark_num(text):
            reward = 0
            # 只要出现了对应的标签，就给予部分奖励
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward
        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

```

### 3. 内容语义奖励与加权 (Semantic Content Reward)

最后，我们使用外部的 Reward Model对内容的实质质量进行打分。

对于推理模型，我们采用了一种特殊的加权策略：

1. 计算**全段回复**（Prompt + Think + Answer）的得分。
2. 提取 `<answer>` 标签内的**纯回答内容**，再次计算得分。
3. 加权得到最终得分 

这种加权机制（0.4/0.6）稍微偏向于最终答案的准确性，同时也兼顾了思考过程的合理性。

```python
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            #Reward Model评价的是对话，而不仅仅是一个句子。例如如果用户要求输出一句有语病的句子，如果仅仅是对句子评价，那么Reward Model会给这个语病句子给出很低的分数，但是这满足了用户的需求，所以应该给一个很高的分数

            #对回答进行解包，解析成Chat Format格式
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            #Reward Model进行打分
            score = reward_model.get_score(reward_tokenizer, tmp_chat)
            #奖励截断，将奖励分数控制在[-3.0, 3.0]这个区间
            scale = 3.0
            score = max(min(score, scale), -scale)

            # 当args.reasoning=1时，额外计算<answer>内容的奖励
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 对answer内容单独计算reward
                    # 伪造一个对话，仅仅包含答案，即假装模型没有废话，直接回答答案
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    # 对这个仅包括答案的对话进行打分，并进行截断
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 对分数进行加权混合处理
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


```

## PPO 训练循环核心 (PPO Training Loop)

### 第一步：环境设置与数据生成 (Rollout Phase)

这一步是 RL 的“探索”阶段。Actor 模型基于当前的 Prompt 生成回复（采样），这个过程不计算梯度（`no_grad`），主要目的是获取“经验数据”。

```python
def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        # 1. 编码 Prompt
        # 注意：这里使用 padding=True 和 truncation=True 确保 batch 内维度一致 (截断和补齐操作)
        # enc 是一个 BatchEncoding 对象, 其含有两个最核心的成员:
        # input_ids:[Batch_Size, Prompt_Length], 将单词/字符映射为词表中的索引ID
        # attention_mask:[Batch_Size, Prompt_Length], 
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                       max_length=args.max_seq_len).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        
        # 使用标量记录 Prompt 的实际长度, 因为经过左侧填充, 所有Prompt_Length都一致
        prompt_length = enc.input_ids.shape[1]

        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法，这是 PyTorch DDP 的特性
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            
            # temperature=0.8 增加了一定的随机性，防止策略过早收敛到局部最优。
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)  # [B, P+R], 即原始模型长度+Response Length, 即模型新生成的回答, 包含了完整的对话内容

```

### 第二步：奖励计算与优势估计 (Reward & Advantage)

获得生成的文本后，我们需要评价它好不好（计算 Reward），并使用 Critic 模型预估当前状态的价值（Value），进而计算优势函数（Advantage）。

```python

        # 从生成的 token 中切片提取出 Response 部分进行解码
        # 这里的len(prompts)就是Batch_size
        responses_text = [tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        
        # 调用之前定义的 calculate_rewards，包含格式奖励、标记奖励和 Reward Model 打分
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]

        # 构建全序列的 mask，因为 Critic 需要看到完整的 input_ids
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]的矩阵，不是Padding的地方为1，是为0
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]，对于每一个位置，都给出了一个标量分数
        
        # 因为arange是递增序列，所以最大的就是最后一个有效Token的位置
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        
        # 提取最后一个 token 对应的 value 作为当前生成的整体价值预估
        # arange生成一个[0, 1, ..., B-1]的行坐标, 再根据last_indices的纵坐标, 提取出最后一个有效Token的索引
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]
        
        # Advantage = 实际获得的奖励 - Critic 预估的价值
        # .detach() 很关键：计算 Advantage 时不反向传播梯度给 Critic
        advantages = rewards - values.detach()  # [B]

```

### 第三步：计算当前策略与参考策略的概率 (Log Probabilities)

这是 PPO 中最“重”计算的一步。我们需要分别计算三个模型（当前 Actor、旧 Actor、参考 Ref）对同一条生成序列的对数概率。

```python

        # 前向传播获取 logits
        # 拿已经生成的回答和输入前向传播一遍，就得到了词表中各个token的"分数"，我们后续要用到
        logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1] (Labels 是 input 向后错一位)
        
        # 获取生成 token 对应的 log_softmax 值
        # [:, :-1]要去掉最后一个token, gather取出输出token的分数，再降维
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
        
        # 创建 mask 以屏蔽 Prompt 部分（我们只关心 Response 的概率）和 Padding 部分
        seq_len = gen_out.size(1) - 1
        # 使用标量 prompt_length 进行比较
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
        # 找到是Padding的部分, 如果为True，必定不是Padding且是Response
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        
        # 求和得到整句 Response 的 log probability
        # 加是因为log(a*b)=log(a)+log(b)
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        # 8. 计算 Old Actor 和 Ref Model 的 Log Prob (不计算梯度)
        with torch.no_grad():
            # Old Actor: 用于计算 PPO 的概率比率 (Ratio)
            # 和上述同样的原理
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            # Reference Model: 用于计算 KL 散度惩罚，防止模型跑偏
            # 和上述同样的原理
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

```

### 第四步：构建 PPO 损失函数 (Loss Calculation)

这里实现了 PPO 论文的核心公式：<div style="overflow-x: auto; padding: 10px;">
$$
\mathcal{L}_{total}(\theta, \phi) = \frac{1}{B} \sum_{i=1}^{B} \left( - \min \left( r_i(\theta) \hat{A}_i, \ \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) + \lambda_{vf} (V_\phi(x_i, y_i) - R_i)^2 + \lambda_{kl} \log \frac{\pi_\theta(y_i|x_i)}{\pi_{ref}(y_i|x_i)} \right)
$$
</div>

```python
        # E[log(p)-log(q)], mean()即代表取平均，也就是E期望
        # 这里是整句话的KL散度, 而不是一个Token的KL散度
        kl = (actor_logp - old_logp).mean()  
        kl_ref = (actor_logp - ref_logp).mean()  
        
        # ratio = exp(log(new) - log(old)) = new / old
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        # 未截断的损失
        surr1 = ratio * advantages  # [B] 

        # 截断 ratio 在 [1-epsilon, 1+epsilon] 之间
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        
        # Policy Loss: 取最小值的负数（因为是梯度下降，要最大化目标函数）
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        
        # Value Loss: MSE Loss，让 Critic 更准
        value_loss = F.mse_loss(values, rewards)  # scalar
        
        # 12. 总 Loss
        # Loss = Policy Loss + c1 * Value Loss + c2 * KL Penalty
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref  # scalar
        loss.backward()

```

### 第五步：反向传播与参数更新 (Optimization)

标准的 PyTorch 优化步骤，包含梯度裁剪（防止梯度爆炸）和梯度累积。

```python
        # 梯度更新 (支持梯度累积)
        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            torch.cuda.empty_cache() # 清理显存碎片

```

### 第六步：日志记录与模型维护 (Logging & Maintenance)

训练循环的最后部分，负责向 WandB 发送数据，定期同步 Old Actor，并保存模型权重。

```python
        # 日志记录
        if is_main_process():
            # 计算平均生成长度，用于监控模型是否出现“沉默”或“啰嗦”倾向
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            # 判断样本是不是真的有<eos>
            has_eos = is_eos.any(dim=1)
            # 计算样本实际生成的有效长度
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            #计算当前batch中样本生成长度的平均值
            avg_len = lengths.float().mean()

            # 提取 scalar 值以便打印
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']
            # wandb用于画图
            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })
            # 日志信息
            Logger(f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                   f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                   f"Reward: {reward_val:.6f}, KL: {kl_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}")

        # Actor模型的延迟更新, 每update_old_actor_freq才更新一次
        if (step + 1) % args.update_old_actor_freq == 0:
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # 保存模型权重
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换为eval模型, 会关闭 Dropout 层，并固定 BatchNorm 的统计量，确保保存的参数是稳定的。
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            actor_state = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            torch.save({k: v.half() for k, v in actor_state.items()}, ckp)
            
            # 保存恢复训练所需要的一切
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            # 重新回到训练模式
            actor_model.train()

```

以下是将 `train_ppo.py` 主程序入口部分（`if __name__ == "__main__":`）按照代码中的注释块进行的 Markdown 分块解析：

### 参数解析与全局配置

这一部分定义了脚本运行所需的所有超参数，包括学习率、模型参数、路径配置以及 PPO 特有的系数（如 `clip_epsilon`, `kl_coef`）。

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    args = parser.parse_args()

```
## 主程序入口
### 1. 初始化环境和随机种子

初始化分布式训练环境（如果需要），并设置随机种子以确保实验的可复现性。

```python
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

```

### 2. 配置目录、模型参数、检查ckp

创建保存目录，实例化模型配置对象 `MiniMindConfig`。如果开启了 `from_resume`，则会尝试寻找之前保存的 checkpoint 信息。

```python
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

```

### 3. 设置混合精度

根据设备类型和参数设置自动混合精度上下文（AMP），通常使用 `bfloat16` 或 `float16` 以节省显存并加速训练。

```python
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

```

### 4. 配置wandb

初始化 Weights & Biases (WandB) 或 SwanLab 用于实验监控。如果从断点恢复，会尝试恢复对应的 run ID。

```python
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

```

### 5. 初始化模型和数据

这是最关键的初始化步骤，构建了 PPO 所需的四个模型：

1. **Actor Model**: 当前训练的策略网络。
2. **Old Actor**: 用于计算比率（Ratio）的旧策略网络（冻结参数）。
3. **Reference Model**: 用于计算 KL 散度的参考网络（冻结参数）。
4. **Critic Model**: 价值网络，用于估计状态价值。
此外，还加载了 **Reward Model**、数据集和优化器。

```python
    # 加载模型权重
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    # Actor模型
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # 生成式任务中，因为生成的Token都是向右追加的，所以要左对齐
    tokenizer.padding_side = 'left'  
    # Old Actor模型
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    # 冻结参数，不参与训练
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Critic模型
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # Reward模型
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    # 数据和优化器
    # 加载提示词数据集
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 优化器配置
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    # 学习率调度与初始化
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

```

### 6. 从ckp恢复状态

如果检测到 checkpoint 数据，将所有模型（Actor, Critic）、优化器和调度器的状态恢复到之前保存的点。

```python
    # 从checkpoint恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

```

### 7. DDP 分布式封装

如果是分布式训练，使用 `DistributedDataParallel` (DDP) 封装 Actor 和 Critic 模型，同时忽略特定的 MoE 参数（如 `freqs_cos`）以避免广播错误。

```python
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)

```

### 8. 开始训练

进入主训练循环。这里处理了断点续训时的数据加载器跳过逻辑（`SkipBatchSampler`），并调用核心函数 `ppo_train_epoch` 开始 PPO 的 Epoch 训练。

```python
    # 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + start_step + 1, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)

```


> "千里之行，始于足下。"