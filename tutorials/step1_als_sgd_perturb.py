#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
TUTORIAL: 一次完整的 ALS → SGD → Perturb 后训练流程
═══════════════════════════════════════════════════════════════════════════════

这个脚本从零开始：
  1. 下载模型 (GPT-2, 124M 参数)
  2. 下载数据 (WikiText-2)
  3. 完整执行一个 ALS → SGD → Perturb 周期
  4. 在每一步打印可读的输出，让你理解内部发生了什么

运行：
  cd /home/room115/alternating-optimization-lora
  python tutorials/step1_als_sgd_perturb.py

预计运行时间：~5 分钟 (GPU) / ~15 分钟 (CPU)
GPU 内存需求：~2 GB
首次运行需下载模型 (~500MB) 和数据 (~5MB)，需要联网。
"""

import math
import time
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# 第 1 步：下载模型
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("第 1 步：下载 GPT-2 模型 (124M 参数)")
print("=" * 70)

MODEL_NAME = "gpt2"  # 最小的 GPT 模型，124M 参数，12 层

# 选择设备：优先 GPU，否则 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  使用设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 这一步会从 HuggingFace 下载 GPT-2 到本地缓存 (~500MB)
print("  正在下载 GPT-2 模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=None,
).to(DEVICE)

# 打印模型结构
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  模型参数总数: {total_params:,} (可训练: {trainable_params:,})")
print(f"  模型层数: {model.config.n_layer}")
print(f"  隐藏维度: {model.config.n_embd}")
print(f"  词表大小: {model.config.vocab_size}")

# 找到 lm_head（输出投影层）— 这是 ALS 要修改的目标
lm_head = None
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and "lm_head" in name:
        lm_head = module
        print(f"  找到 lm_head 层: '{name}', 形状={list(module.weight.shape)}")
        break

# ══════════════════════════════════════════════════════════════════════════════
# 第 2 步：下载数据 & 构建 DataLoader
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("第 2 步：下载 WikiText-2 数据集")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 没有 pad_token，用 eos_token 代替

# 下载训练集
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# 下载测试集
eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

print(f"  训练样本数: {len(train_dataset)}")
print(f"  测试样本数: {len(eval_dataset)}")


def tokenize(examples):
    """将文本转为 token IDs。"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,       # 使用较短序列以加快演示
        padding="max_length",
    )


# Tokenize 两个数据集
train_tokenized = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
eval_tokenized = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])
train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 构建 DataLoader
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4

# collate_fn: 自动构造 labels = input_ids.clone()（因果语言模型的标准做法）
def collate_with_labels(batch):
    result = {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
    result["labels"] = result["input_ids"].clone()
    return result

train_dl = DataLoader(train_tokenized, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                       collate_fn=collate_with_labels)
eval_dl = DataLoader(eval_tokenized, batch_size=EVAL_BATCH_SIZE,
                      collate_fn=collate_with_labels)

# 取一个 batch 看看形状
sample_batch = next(iter(train_dl))
print(f"  单个 batch 形状: {sample_batch['input_ids'].shape}")
print(f"  (batch_size={TRAIN_BATCH_SIZE}, seq_len=128)")

# ══════════════════════════════════════════════════════════════════════════════
# 第 3 步：评估函数（训练前基线）
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("第 3 步：训练前基线评估")
print("=" * 70)


def evaluate(model: nn.Module, eval_loader: DataLoader) -> dict:
    """
    计算模型的困惑度 (Perplexity)。

    困惑度 = exp(交叉熵损失)
    难度越低 → PPL 越低 → 模型越好。
    原始 GPT-2 在 WikiText-2 上的 PPL 约为 30-40。
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            # HuggingFace 模型：传 labels=input_ids 才会自动计算 loss
            batch["labels"] = batch["input_ids"].clone()
            outputs = model(**batch)
            loss = outputs.loss
            # 加权：每个 token 的贡献 = loss × token 数量
            mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
            n_tokens = mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    model.train()
    return {"loss": avg_loss, "perplexity": perplexity, "n_tokens": total_tokens}


baseline = evaluate(model, eval_dl)
print(f"  未训练基线: loss={baseline['loss']:.4f}, ppl={baseline['perplexity']:.2f}")
print(f"  评估 token 数: {baseline['n_tokens']:,}")

# ══════════════════════════════════════════════════════════════════════════════
# 第 4 步：定义训练的三个阶段
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("第 4 步：定义 ALS / SGD / Perturb 三个阶段")
print("=" * 70)

# ── 超参数 ──
ALS_BLOCK_SIZE = 1024     # ALS 每次求解的行数
ALS_REG_LAMBDA = 1e-3     # ALS 正则化强度（防止病态矩阵）
ALS_STEP_SIZE = 0.01      # EMA 混合系数（1% 新解, 99% 旧权重）
SGD_LR = 1e-4             # SGD 学习率
SGD_MOMENTUM = 0.9        # SGD 动量
SGD_WEIGHT_DECAY = 0.01   # SGD 权重衰减 (L2 正则化)
PERTURB_SCALE = 1e-3      # 扰动噪声标准差
SGD_STEPS_PER_CYCLE = 50  # 每个周期内 SGD 步数


def als_phase(model, lm_head_module, batch, block_size, reg_lambda, step_size):
    """
    ┌───────────────────────────────────────────────────────────────────┐
    │ ALS 阶段：对 lm_head 做块状最小二乘求解                             │
    │                                                                   │
    │ 数学: W_new = (XᵀX + λI)⁻¹ Xᵀ Y_target                          │
    │                                                                   │
    │ 流程:                                                             │
    │  1. 通过 forward hook 捕获 lm_head 的输入激活 X                   │
    │  2. 对每个 1024 行 block:                                         │
    │     a. 找出标签落在该 block 内的 token                            │
    │     b. 构造 Y_target (one-hot 目标矩阵)                           │
    │     c. 用 Cholesky 分解解 W_new = (XᵀX+λI)⁻¹ Xᵀ Y_target       │
    │     d. EMA 更新: W ← 0.99×W + 0.01×W_new                         │
    │  3. 返回平均重构损失                                               │
    └───────────────────────────────────────────────────────────────────┘
    """
    weight = lm_head_module.weight.data  # [vocab_size, d_model]
    vocab_size, d_model = weight.shape
    device = weight.device
    labels = batch.get("labels")

    if labels is None:
        print("  ALS: 没有 labels，跳过")
        return 0.0

    # ── 步骤 A1: 通过 hook 捕获输入激活 ──
    activations: list[torch.Tensor] = []
    hook = lm_head_module.register_forward_pre_hook(
        lambda _mod, inp: activations.append(inp[0].detach())
    )

    # 前向传播：收集 lm_head 的输入
    with torch.no_grad():
        batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        _ = model(**batch_gpu)
    hook.remove()

    if not activations:
        print("  ALS: 未捕获到激活")
        return 0.0

    X = activations[0]  # [N, d_model]，可能是 3D → 压平
    if X.dim() == 3:
        X = X.reshape(-1, d_model)

    N = X.shape[0]
    labels_flat = labels.reshape(-1)[:N].to(device=device, dtype=torch.long)
    labels_flat = torch.clamp(labels_flat, 0, vocab_size - 1)

    # ── 步骤 A2: 预计算 XᵀX + λI（所有 block 共享） ──
    X_f32 = X.detach().float()
    XtX = X_f32.T @ X_f32                                    # [d_model, d_model]
    reg = reg_lambda * torch.eye(d_model, device=device, dtype=torch.float32)
    XtX_reg = XtX + reg

    try:
        L = torch.linalg.cholesky(XtX_reg)
    except RuntimeError:
        L = None  # 矩阵不正定时退化为最小二乘

    # 保存旧权重用于范数检查
    weight_old = weight.detach().clone().float()
    n_blocks = (vocab_size + block_size - 1) // block_size
    total_loss = 0.0

    # ── 步骤 A3: 分块求解 ──
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, vocab_size)

        # 找到标签落在 [start, end) 范围内的 token
        mask = (labels_flat >= start) & (labels_flat < end)
        if not mask.any():
            continue

        X_masked = X_f32[mask]          # [n_masked, d_model]
        target_tokens = labels_flat[mask] - start

        # 构造 one-hot Y_target
        n_masked = mask.sum().item()
        Y_target = torch.zeros(n_masked, end - start, device=device, dtype=torch.float32)
        Y_target[torch.arange(n_masked, device=device), target_tokens] = 1.0

        # 核心公式：W_new = (XᵀX + λI)⁻¹ Xᵀ Y
        XtX_masked = X_masked.T @ X_masked + reg
        XtY = X_masked.T @ Y_target

        try:
            L_masked = torch.linalg.cholesky(XtX_masked)
            W_new_block = torch.cholesky_solve(XtY, L_masked).T  # [block, d_model]
        except RuntimeError:
            W_new_block = torch.linalg.lstsq(XtX_masked, XtY).solution.T

        # ── EMA 阻尼更新：只采纳 step_size 比例的新解 ──
        W_current = weight_old[start:end, :]
        damped = (1 - step_size) * W_current + step_size * W_new_block
        weight[start:end, :] = damped.to(device=device, dtype=weight.dtype)

        # 计算交叉熵损失（用于日志）
        pred = X_masked @ W_new_block.T
        ce = -(Y_target * torch.log_softmax(pred, dim=-1)).sum() / n_masked
        total_loss += ce.item()

    # ── 步骤 A4: 检查权重变化是否过大 ──
    delta = weight - weight_old
    delta_norm = torch.norm(delta.float()).item()
    old_norm = torch.norm(weight_old.float()).item()
    if old_norm > 1e-12:
        change_ratio = delta_norm / old_norm
        print(f"  ALS: 解决了 {n_blocks} blocks, ‖ΔW‖/‖W‖={change_ratio:.6f}")
    else:
        print(f"  ALS: 解决了 {n_blocks} blocks")

    return total_loss / max(n_blocks, 1)


def sgd_phase(model, batch, lr, momentum, weight_decay, optimizer):
    """
    ┌───────────────────────────────────────────────────────────────────┐
    │ SGD 阶段：标准随机梯度下降                                         │
    │                                                                   │
    │ 流程:                                                             │
    │  1. optimizer.zero_grad()  — 清空梯度                             │
    │  2. model(**batch)         — 前向传播 → 计算 loss                  │
    │  3. loss.backward()        — 反向传播 → 计算梯度                   │
    │  4. clip_grad_norm_()      — 梯度裁剪 (防止爆炸)                   │
    │  5. optimizer.step()       — 更新参数                              │
    └───────────────────────────────────────────────────────────────────┘
    """
    optimizer.zero_grad()

    device = next(model.parameters()).device
    batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    # HuggingFace causal LM 模型只在传入 labels 时才计算 loss
    batch_gpu["labels"] = batch_gpu["input_ids"].clone()
    outputs = model(**batch_gpu)
    loss = outputs.loss

    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(
        filter(lambda p: p.requires_grad, model.parameters()),
        max_norm=1.0,
    )

    optimizer.step()
    return loss.item()


def perturb_phase(model, scale):
    """
    ┌───────────────────────────────────────────────────────────────────┐
    │ Perturb 阶段：参数空间随机扰动                                     │
    │                                                                   │
    │ 对每个可训练参数: θ ← θ + ε, ε ~ N(0, σ²)                         │
    │                                                                   │
    │ 目的：                                                            │
    │  - 把参数推出尖锐的局部极小值                                       │
    │  - 改善泛化 (SAM / RWP 效果)                                      │
    │                                                                   │
    │ 不同层类型用不同的缩放因子：                                        │
    │  - embedding 层: 10%（语义信息敏感）                               │
    │  - attention 层: 50%                                              │
    │  - FFN 层: 100%（冗余度高）                                       │
    └───────────────────────────────────────────────────────────────────┘
    """
    total_energy = 0.0
    n_params_total = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # 按层类型缩放噪声
            if "embed" in name.lower():
                layer_scale = scale * 0.1
            elif "attn" in name.lower() or "attention" in name.lower():
                layer_scale = scale * 0.5
            elif "ffn" in name.lower() or "mlp" in name.lower():
                layer_scale = scale * 1.0
            else:
                layer_scale = scale * 0.5

            noise = torch.randn_like(param) * layer_scale
            param.add_(noise)

            total_energy += (noise ** 2).sum().item()
            n_params_total += param.numel()

    avg_energy = total_energy / max(n_params_total, 1)
    return avg_energy


# ══════════════════════════════════════════════════════════════════════════════
# 第 5 步：执行一个完整的 ALS → SGD(50) → Perturb 周期
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("第 5 步：执行 ALS → SGD(50) → Perturb 一个完整周期")
print("=" * 70)

# 创建 SGD 优化器
sgd_optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=SGD_LR,
    momentum=SGD_MOMENTUM,
    weight_decay=SGD_WEIGHT_DECAY,
)

model.train()

step = 0
loss_history: list[dict] = []
eval_history: list[dict] = []

# ── 5.1 ALS 阶段 ──
print("\n--- 阶段 A: ALS ---")
step += 1

# 用第一个训练 batch 做 ALS
als_batch = next(iter(train_dl))
als_loss = als_phase(model, lm_head, als_batch,
                      block_size=ALS_BLOCK_SIZE,
                      reg_lambda=ALS_REG_LAMBDA,
                      step_size=ALS_STEP_SIZE)
loss_history.append({"step": step, "phase": "ALS", "loss": als_loss})
print(f"  step {step:3d} | ALS   | loss = {als_loss:10.2f}")


# ── 5.2 SGD 阶段 ──
print(f"\n--- 阶段 B: SGD ({SGD_STEPS_PER_CYCLE} 步) ---")

for _ in range(SGD_STEPS_PER_CYCLE):
    step += 1
    sgd_batch = next(iter(train_dl))
    sgd_loss = sgd_phase(model, sgd_batch,
                          lr=SGD_LR,
                          momentum=SGD_MOMENTUM,
                          weight_decay=SGD_WEIGHT_DECAY,
                          optimizer=sgd_optimizer)
    loss_history.append({"step": step, "phase": "SGD", "loss": sgd_loss})

    # 每 10 步打印一次
    if step % 10 == 0:
        print(f"  step {step:3d} | SGD   | loss = {sgd_loss:10.4f}")


# ── 5.3 Perturb 阶段 ──
print(f"\n--- 阶段 C: Perturb ---")
step += 1

perturb_energy = perturb_phase(model, scale=PERTURB_SCALE)
loss_history.append({"step": step, "phase": "PERTURB", "loss": perturb_energy})
print(f"  step {step:3d} | PERT  | noise_energy = {perturb_energy:.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# 第 6 步：训练后评估
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("第 6 步：训练后评估")
print("=" * 70)

after = evaluate(model, eval_dl)
print(f"  训练前: loss={baseline['loss']:.4f}, ppl={baseline['perplexity']:.2f}")
print(f"  训练后: loss={after['loss']:.4f}, ppl={after['perplexity']:.2f}")
print(f"  变化:   loss Δ={after['loss'] - baseline['loss']:+.4f}, "
      f"ppl Δ={after['perplexity'] - baseline['perplexity']:+.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 第 7 步：总结
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("第 7 步：总结 — 你刚才做了什么")
print("=" * 70)

print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                  一次完整的 ALS→SGD→Perturb 后训练周期           │
  │                                                                 │
  │  ALS (1步)  │  对 lm_head 的 50257×768 权重矩阵做块状最小二乘    │
  │             │  W_new = (XᵀX + λI)⁻¹ Xᵀ Y                      │
  │             │  分 ~50 个 block 独立求解，EMA 阻尼只采纳 1%        │
  │             │                                                    │
  │  SGD (50步) │  标准随机梯度下降消化 ALS 引入的参数变化            │
  │             │  lr=1e-4, momentum=0.9, weight_decay=0.01          │
  │             │  全部 124M 参数通过梯度反向传播同步调整              │
  │             │                                                    │
  │  Perturb(1步)│  对全部参数注入高斯噪声 N(0, 1e-6)                  │
  │             │  不同层类型用不同缩放因子                           │
  │             │  embedding 层保护 (0.1×), FFN 层全噪声 (1.0×)       │
  │                                                                 │
  │  结果       │  PPL: {before:>6.2f} → {after:>6.2f}                        │
  └─────────────────────────────────────────────────────────────────┘
""".format(before=baseline['perplexity'], after=after['perplexity']))

print("下一步:")
print("  1. 修改 SGD_STEPS_PER_CYCLE 跑更长的消化期")
print("  2. 添加多个 ALS→SGD→Perturb 周期观察非单调收敛")
print("  3. 换用更大的模型 (OPT-125m, Qwen2.5-0.5B)")
print("  4. 在 tutorials/step2_multi_cycle.py 中继续")
