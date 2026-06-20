# Torchrun × 2 DeepSpeed Protocol A — Design Spec

**Date**: 2026-06-20
**Status**: Draft
**Target**: Qwen2.5-7B Protocol A (AltOpt + Full-Rank) via torchrun × 2 + DeepSpeed ZeRO-2

---

## 1. Problem Statement

Protocol A (AltOpt + Full-Rank) is the last missing cell in the 2×2 factorial matrix
for Qwen2.5-7B.  Six single-process DeepSpeed attempts failed because:

1. Single-process ZeRO-2 does NOT shard model parameters across GPUs — the full 28GB
   fp32 model copy on GPU 0 exceeds 32GB before initialization completes.
2. Intermediate-layer reconstruction ALS produces hallucinated solutions
   (‖ΔW‖/‖W‖ > 10⁶) on ≥28-layer models — only lm_head ALS is safe.
3. PyTorch SGD optimizer is rejected by DeepSpeed CPU-offload mode.

**Solution**: Use `torchrun --nproc_per_node=2` for true multi-process ZeRO-2 sharding,
and modify `AltOptTrainer._train_deepspeed()` to correctly route ALS/SGD/Perturb
through the DeepSpeed engine.

---

## 2. Architecture Changes

### 2.1 Current Broken Path

```
AltOptFramework.step(batch)
  → _execute_phase(Phase.ALS)
    → ALSBlockSolver.solve_block(batch)
      → model(**batch)          # raw forward, no DeepSpeed
      → cholesky + weight update # raw weight mutation
  → _execute_phase(Phase.SGD)
    → SGDPhaseOptimizer.step(batch)
      → model(**batch)          # raw forward
      → loss.backward()         # raw backward
      → optimizer.step()        # raw SGD step
```

The entire path bypasses DeepSpeed — no mixed precision, no gradient sharding,
no ZeRO communication.  This works on small CPU models but breaks on 7B GPU.

### 2.2 Required Path

```
trainer._train_deepspeed(dataloader)
  for batch in dataloader:
    batch_gpu = {k: v.to(engine.device) for ...}

    if phase == ALS:
      loss = _deepspeed_als_step(engine, self.altopt.als, batch_gpu)
    elif phase == SGD:
      loss = _deepspeed_sgd_step(engine, self.altopt, batch_gpu)
    elif phase == PERTURB:
      loss = _deepspeed_perturb_step(engine, self.altopt.perturb)

    self._on_step_end(loss)
```

### 2.3 ALS through DeepSpeed

```
_deepspeed_als_step(engine, als_solver, batch):
  # 1. Forward pass through DeepSpeed engine (bf16 mixed precision)
  outputs = engine(batch['input_ids'], attention_mask=batch['attention_mask'])
  
  # 2. Create a fresh forward to collect lm_head input activations
  #    (engine.module is the unwrapped model on this rank)
  with torch.no_grad():
      activations = []
      hook = engine.module.lm_head.register_forward_pre_hook(
          lambda mod, inp: activations.append(inp[0].detach()))
      _ = engine.module(**batch)  # second forward, no grad
      hook.remove()
  
  # 3. Only rank 0 solves ALS (Cholesky is not collective)
  if torch.distributed.get_rank() == 0:
      X = activations[0]  # [N, d_model] bf16
      X_f32 = X.float()
      # ... Cholesky solve (same as _solve_head_layer but on rank 0) ...
      W_new = ...  # [vocab_size, d_model] float32
      # EMA damp + copy to lm_head weight
      engine.module.lm_head.weight.data.copy_(W_new.to(weight.dtype))
  
  # 4. Broadcast the updated lm_head weight to all ranks
  torch.distributed.broadcast(
      engine.module.lm_head.weight.data, src=0)
  
  return recon_loss
```

**Key design decisions:**
- ALS only touches lm_head (confirmed safe on 28L).
- Only rank 0 runs Cholesky; result broadcast to rank 1.
- Two forward passes per ALS step: one for the loss signal, one with hooks
  for activation capture.  This is acceptable because ALS runs only 1 step
  per cycle.
- Weight update is in-place on `engine.module` — ZeRO-2 keeps all-gathered
  params, so this is safe.

### 2.4 SGD through DeepSpeed

```
_deepspeed_sgd_step(engine, altopt, batch):
  # Use DeepSpeed's built-in forward/backward/step pipeline
  # SGD momentum state is sharded via ZeRO-2
  
  outputs = engine(**batch)
  loss = outputs.loss
  
  engine.backward(loss)       # gradient all-reduce + shard
  engine.step()               # optimizer step on sharded grads
  
  altopt.sgd.last_grad_norm = ... # extract for logging
  return loss.item()
```

**Critical**: Use DeepSpeed's own optimizer, NOT PyTorch SGD.
DeepSpeedCPUAdam is the safest choice (no CUDA mismatch JIT).
Set `zero_force_ds_cpu_optimizer: false` to allow custom optimizer
(DeepSpeedCPUAdam is actually a native DS optimizer but the env var
squelches the warning).

### 2.5 Perturb through DeepSpeed

```
_deepspeed_perturb_step(engine, perturb_scheduler, batch):
  # Noise injection: works on engine.module params (all-gathered)
  with torch.no_grad():
      for name, param in engine.module.named_parameters():
          if param.requires_grad:
              noise = torch.randn_like(param) * noise_scale
              param.add_(noise)
  
  # One forward to measure loss for logging
  with torch.no_grad():
      outputs = engine(**batch)
  
  return outputs.loss.item()
```

---

## 3. Modifications to Existing Code

### 3.1 `altopt/trainer.py`

**`_setup_deepspeed()`**: Accept and configure DeepSpeedCPUAdam for SGD.
Current code already does this for Protocol B — extend to Protocol A.

```python
def _setup_deepspeed(self):
    from .deepspeed_engine import DeepSpeedConfig, DeepSpeedEngine
    
    ds_cfg = DeepSpeedConfig(
        zero_stage=2,
        bf16_enabled=True,
        gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        train_micro_batch_size_per_gpu=1,
    )
    
    # DeepSpeedCPUAdam for SGD momentum (ZeRO-2 compatible)
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    self.optimizer = DeepSpeedCPUAdam(
        filter(lambda p: p.requires_grad, self.model.parameters()),
        lr=self.config.lr, betas=(0.9, 0.999),
        weight_decay=self.config.weight_decay,
    )
    
    self._deepspeed_engine = DeepSpeedEngine(
        self.model, ds_cfg, optimizer=self.optimizer,
    )
    self._deepspeed_initialized = False
    
    # Initialize AltOpt framework AFTER DeepSpeed engine creation
    # (AltOpt wraps the model that DS will manage)
    schedule = self.config.phase_schedule or ...
    self.altopt = AltOptFramework(self.model, schedule)
    self._has_altopt = True
```

**`_train_deepspeed()`**: Rewrite AltOpt branch to route through engine:

```python
def _train_deepspeed(self, dataloader):
    if not self._deepspeed_initialized:
        self._deepspeed_engine.initialize(dataloader)
        self._deepspeed_initialized = True
    
    engine = self._deepspeed_engine.engine
    self.model.train()
    cfg = self.config
    
    for epoch in range(cfg.max_epochs or 1):
        for batch in dataloader:
            self._on_step_start(batch)
            
            device = engine.device
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            
            if self._has_altopt:
                loss = self._altopt_deepspeed_step(engine, batch_gpu)
            else:
                # Protocol B path (unchanged)
                outputs = engine(**batch_gpu)
                raw_loss = outputs.loss
                engine.backward(raw_loss)
                engine.step()
                loss = raw_loss.item()
            
            self._on_step_end(loss)
            self.state.step += 1
            
            if cfg.max_steps and self.state.step >= cfg.max_steps:
                return self.state
    
    return self.state

def _altopt_deepspeed_step(self, engine, batch):
    """Route AltOpt phases through DeepSpeed engine."""
    altopt = self.altopt
    altopt._ensure_phase()
    config = altopt.schedule.phases[altopt._phase_index]
    
    if config.phase == Phase.ALS:
        return self._deepspeed_als_step(engine, batch, config)
    elif config.phase == Phase.SGD:
        return self._deepspeed_sgd_step(engine, batch, config)
    elif config.phase == Phase.PERTURB:
        return self._deepspeed_perturb_step(engine, batch, config)
```

Add three new methods:
- `_deepspeed_als_step(engine, batch, config)` — lm_head ALS + broadcast
- `_deepspeed_sgd_step(engine, batch, config)` — DeepSpeed forward/backward/step
- `_deepspeed_perturb_step(engine, batch, config)` — noise injection

### 3.2 `altopt/framework.py`

Add `_ensure_phase()` method so the trainer can query current phase
without calling `step()`:

```python
def _ensure_phase(self):
    """Advance phase index if needed without executing a step."""
    if self._phase_index >= len(self.schedule.phases):
        self._cycle_count += 1
        if self._cycle_count >= self.schedule.cycles:
            return  # all cycles complete
        self._phase_index = 0
    
    config = self.schedule.phases[self._phase_index]
    self.state.current_phase = config.phase
    self.state.current_cycle = self._cycle_count
```

### 3.3 `altopt/deepspeed_engine.py`

Ensure `to_dict()` includes all required fields:
- `zero_allow_untested_optimizer: True`
- `zero_force_ds_cpu_optimizer: False`
- `train_micro_batch_size_per_gpu: 1`
- `gradient_accumulation_steps`: from config
- `bf16: {enabled: True}`
- `zero_optimization: {stage: 2, offload_optimizer: {device: cpu}, ...}`

### 3.4 `experiments/run_7b_gpu.py`

**Schedule for Protocol A** (800 steps budget):
```python
schedule = PhaseSchedule(
    phases=[
        PhaseConfig(phase=Phase.ALS, steps=1, block_size=512),
        PhaseConfig(phase=Phase.SGD, steps=350, lr=5e-5),
        PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=5e-4),
    ],
    cycles=2)  # 2 × 352 = 704 steps
```

**DeepSpeed config override:**
```python
DEEPSPEED_CFG = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16,
    "bf16": {"enabled": True},
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        ...
    },
}
```

---

## 4. Launch Script

```bash
#!/usr/bin/env bash
# Protocol A: AltOpt + Full-Rank, torchrun × 2 + DeepSpeed ZeRO-2

export LD_LIBRARY_PATH="<nvidia lib dirs>"
export PATH="$VENV/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export DS_SKIP_CUDA_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJ_DIR"

for seed in 42 123 456; do
    torchrun --nproc_per_node=2 --master_port=$((29500 + seed)) \
        -m experiments.run_7b_gpu_torchrun A altopt full_rank $seed 800
done
```

**Run script wrapper** (`experiments/run_7b_gpu_torchrun.py`):
A small `__main__` entry point that parses CLI args and calls `run_protocol()`.

---

## 5. Risk Items & Contingencies

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|------------|
| ALS broadcast hangs (NCCL timeout) | Medium | Critical | Add barrier before broadcast, set NCCL_TIMEOUT=600 |
| OOM during ALS forward (2 forwards) | Medium | High | Reduce block_size, skip 2nd forward if possible |
| DeepSpeedCPUAdam JIT fails | Low | High | Already proven working in Protocol B |
| Rank-0 Cholesky OOM on lm_head (152064×3584) | Low | High | bf16 intermediate, block-wise solve |
| NaN in SGD after ALS update | Medium | High | ALS norm check already in place, rollback on catastrophic |

### Contingency: If torchrun fails entirely

Fall back to single-GPU SGD+Perturb Protocol A':
```
A' = SGD(200s) + Perturb(1s) × 4 cycles, no ALS, no DeepSpeed
```
No ALS means no Cholesky, no broadcast, no rank coordination needed.
Can run on single GPU with device_map="auto" + batch_size=1.
Still provides the "optimizer effect on full-rank" comparison vs Protocol B.
This is a scientifically valid ablation — it tests whether the alternating
mechanism (SGD+Perturb) differs from continuous AdamW at full-rank.

---

## 6. Validation Checklist

Before running full 3-seed experiment:

- [ ] `torchrun --nproc_per_node=2` launches without MPI errors
- [ ] DeepSpeed initializes with ZeRO-2 on both ranks
- [ ] ALS step completes without hang (both ranks see broadcast)
- [ ] ALS reconstruction loss is finite and reasonable
- [ ] SGD step uses engine.backward/step correctly (no raw .backward())
- [ ] Loss decreases over first 10 steps
- [ ] No NaN/Inf in first 100 steps
- [ ] GPU memory stable across 100 steps (no leak)
- [ ] Single seed produces PPL comparable to Protocol B (within 10-100×)

---

## 7. Success Criteria

**Primary**: Protocol A × 3 seeds complete without NaN/OOM on Qwen2.5-7B.

**Secondary**: A-B and (A-B)-(C-D) interaction effect computable at 7B scale.

**Minimum viable**: Any valid PPL value for Protocol A that isn't NaN/Inf.
Even if A >> B (which is expected), the interaction effect can be computed.

---

*Design date: 2026-06-20*
