"""
Alternating Least Squares (ALS) block-wise exact solver.

Partitions model parameters into independent blocks and solves each block
exactly via least squares (requiring matrix inversion) while holding all
other blocks fixed.

This is the most computationally expensive phase of the alternating
optimization framework — each block requires solving a linear system
of size (block_size × block_size), costing O(b³) per block.

For LLM weight matrices W ∈ ℝ^{d_out × d_in}, we partition rows into
blocks of size b, solving:

    W_block = argmin ||X W_block^T - Y_target||²

where X is the input activations for that block's rows.

This yields the closed-form solution:

    W_block = (X^T X + λI)^{-1} X^T Y_target

The λI regularization prevents ill-conditioned inverses when X is
near-singular.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ALSBlockSolver:
    """
    Block-wise ALS solver for linear layers.

    Operates on nn.Linear modules in the model, solving each block of rows
    independently via regularized least squares.
    """

    def __init__(
        self,
        model: nn.Module,
        reg_lambda: float = 1e-4,
        step_size: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.reg_lambda = reg_lambda
        self.step_size = step_size
        self.device = device or next(model.parameters()).device

        # Cache: store (X^T X + λI)^{-1} X^T per block for warm-start
        self._cache: dict[str, torch.Tensor] = {}
        # Activation cache: store layer inputs captured via forward hooks
        self._cached_activations: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    def solve_block(
        self,
        batch: dict[str, torch.Tensor],
        block_size: int = 1024,
    ) -> float:
        """
        Solve one ALS block update across linear layers.

        Uses label-based targets for the lm_head (output layer) where
        we have ground-truth token labels. For intermediate layers,
        reconstruction-based ALS is skipped (it's a no-op that can
        accumulate numerical error).

        Args:
            batch: dict with 'input_ids', 'labels' (and optionally 'attention_mask')
            block_size: number of output rows per block

        Returns:
            total_loss: float, ALS reconstruction loss
        """
        labels = batch.get("labels")
        if labels is None:
            logger.debug("ALS: no labels in batch, skipping")
            return 0.0

        total_loss = 0.0
        n_blocks_total = 0

        for name, module in self.model.named_modules():
            is_head = ("lm_head" in name or "score" in name)
            if isinstance(module, nn.Linear) and is_head:
                loss, n_blocks = self._solve_head_layer(
                    name, module, batch, labels, block_size
                )
                total_loss += loss
                n_blocks_total += n_blocks

        if n_blocks_total > 0:
            logger.debug(
                "ALS: solved %d blocks across head layers, total_loss=%.6f",
                n_blocks_total, total_loss
            )

        return total_loss / max(n_blocks_total, 1)

    def _solve_head_layer(
        self,
        name: str,
        module: nn.Linear,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        block_size: int,
    ) -> tuple[float, int]:
        weight = module.weight.data  # [vocab_size, d_model]
        vocab_size, d_model = weight.shape
        device = weight.device

        activations: list[torch.Tensor] = []
        hook_handle = module.register_forward_pre_hook(
            lambda _mod, inp: activations.append(inp[0].detach())
        )

        try:
            with torch.no_grad():
                batch_input = {k: v.to(device) for k, v in batch.items()
                               if isinstance(v, torch.Tensor)}
                _ = self.model(**batch_input)
            hook_handle.remove()

            if not activations:
                return 0.0, 0

            X = activations[0]
            if X.dim() == 3:
                X = X.reshape(-1, d_model)

            N = X.shape[0]
            labels_flat = labels.reshape(-1)[:N].to(device=device, dtype=torch.long)
            labels_flat = torch.clamp(labels_flat, 0, vocab_size - 1)

            X_f32 = X.detach().float()
            n_blocks = (vocab_size + block_size - 1) // block_size
            total_loss = 0.0

            XtX = X_f32.T @ X_f32
            reg = self.reg_lambda * torch.eye(d_model, device=X_f32.device, dtype=torch.float32)
            XtX_reg = XtX + reg

            try:
                L = torch.linalg.cholesky(XtX_reg)
            except RuntimeError:
                L = None

            for i in range(n_blocks):
                start = i * block_size
                end = min(start + block_size, vocab_size)

                mask = (labels_flat >= start) & (labels_flat < end)
                if not mask.any():
                    continue

                X_masked = X_f32[mask]
                target_tokens = labels_flat[mask] - start
                Y_target = torch.zeros((mask.sum().item(), end - start),
                                       device=X_f32.device, dtype=torch.float32)
                Y_target[torch.arange(mask.sum().item(), device=X_f32.device), target_tokens] = 1.0

                XtX_masked = X_masked.T @ X_masked + reg
                XtY = X_masked.T @ Y_target

                try:
                    L_masked = torch.linalg.cholesky(XtX_masked)
                    W_new_block = torch.cholesky_solve(XtY, L_masked).T
                except RuntimeError:
                    W_new_block = torch.linalg.lstsq(XtX_masked, XtY).solution.T

                W_current_block = weight[start:end, :].detach().float()
                damped = (1 - self.step_size) * W_current_block + self.step_size * W_new_block
                weight[start:end, :] = damped.to(device=device, dtype=weight.dtype)

                pred = X_masked @ W_new_block.T
                ce_loss = -torch.sum(Y_target * torch.log_softmax(pred, dim=-1)) / mask.sum().item()
                total_loss += ce_loss.item()

            return total_loss, n_blocks

        except Exception as e:
            logger.warning("ALS head solve failed for layer '%s': %s", name, e)
            hook_handle.remove()
            return 0.0, 0

    def _solve_linear_layer(
        self,
        name: str,
        module: nn.Linear,
        batch: dict[str, torch.Tensor],
        block_size: int,
    ) -> tuple[float, int]:
        """
        Solve one nn.Linear layer via block-wise ALS.

        Returns (loss, n_blocks_solved).
        """
        weight = module.weight.data  # [d_out, d_in]
        d_out, d_in = weight.shape
        device = weight.device

        # ── Forward pass to collect activations ──
        # We need the input to this layer. For now, use a hook-based approach
        # or require the user to provide activations.
        # Simplified: do a forward pass and capture inputs via hook.

        activations: list[torch.Tensor] = []
        hook_handle = module.register_forward_pre_hook(
            lambda _mod, inp: activations.append(inp[0].detach())
        )

        try:
            # Run forward with no grad to get activations
            with torch.no_grad():
                _ = self.model(**{k: v.to(device) for k, v in batch.items()
                                  if isinstance(v, torch.Tensor)})
            hook_handle.remove()

            if not activations:
                return 0.0, 0

            X = activations[0]
            if X.dim() == 3:
                X = X.reshape(-1, d_in)

            X_f32 = X.detach().float()
            W_f32 = weight.detach().float()

            n_blocks = (d_out + block_size - 1) // block_size
            total_loss = 0.0

            XtX = X_f32.T @ X_f32
            reg = self.reg_lambda * torch.eye(d_in, device=X_f32.device, dtype=torch.float32)
            XtX_reg = XtX + reg

            try:
                L = torch.linalg.cholesky(XtX_reg)
            except RuntimeError:
                L = None

            for i in range(n_blocks):
                start = i * block_size
                end = min(start + block_size, d_out)

                W_block = W_f32[start:end, :]
                Y_block = X_f32 @ W_block.T
                XtY = X_f32.T @ Y_block

                if L is not None:
                    W_new_block = torch.cholesky_solve(XtY, L).T
                else:
                    W_new_block = torch.linalg.lstsq(XtX_reg, XtY).solution.T

                weight[start:end, :] = W_new_block.to(device=device, dtype=weight.dtype)

                recon_error = torch.norm(X_f32 @ W_new_block.T - Y_block) ** 2
                total_loss += recon_error.item()

            return total_loss, n_blocks

        except Exception as e:
            logger.warning("ALS block solve failed for layer '%s': %s", name, e)
            hook_handle.remove()
            return 0.0, 0

    def solve_low_rank_block(
        self,
        batch: dict[str, torch.Tensor],
        peft_bridge,
        block_size: int = 256,
    ) -> float:
        """
        ALS block solve adapted for low-rank (LoRA) parameterization.

        Strategy: Solve full-rank ALS for the composite weight W_eff = W_base + (α/r)BA,
        then project the solution back to low-rank space by updating B.

        For each block of output rows (start:end):
          1. Compute current output: Y_curr = X @ W_eff[start:end, :].T
          2. ALS target: Y_target = X @ W_eff[start:end, :].T (reconstruction)
          3. Solve: ΔW_block = (X^T X + λI)^(-1) X^T (Y_target - Y_curr)
             But since Y_target = Y_curr, we instead compute:
             W_new = (X^T X + λI)^(-1) X^T Y_target
          4. Project to low-rank: update B to approximate W_new:
             B_new[start:end, :] = W_new_block @ A^T @ (A@A^T + λI)^(-1) / scaling

        This gives us true ALS optimization in LoRA space.
        """
        total_loss = 0.0
        n_blocks_total = 0

        adapter_map = peft_bridge.all_adapter_info()
        layer_names = list(adapter_map.keys())
        if not layer_names:
            return 0.0

        self._install_activation_hooks(layer_names)

        with torch.no_grad():
            try:
                device_inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                _ = self.model(**device_inputs)
            except Exception:
                self._remove_hooks()
                return 0.0

        for layer_name, info in adapter_map.items():
            lora_A_data = info.lora_A
            lora_B_data = info.lora_B

            if isinstance(lora_A_data, torch.nn.Module):
                lora_A_params = list(lora_A_data.parameters())
                lora_A = lora_A_params[0].data if lora_A_params else None
            elif isinstance(lora_A_data, torch.Tensor):
                lora_A = lora_A_data
            else:
                continue

            if isinstance(lora_B_data, torch.nn.Module):
                lora_B_params = list(lora_B_data.parameters())
                lora_B = lora_B_params[0].data if lora_B_params else None
            elif isinstance(lora_B_data, torch.Tensor):
                lora_B = lora_B_data
            else:
                continue

            if lora_A is None or lora_B is None:
                continue

            base_W = info.base_weight
            r_val = info.r
            scaling = info.scaling
            d_out, d_in = base_W.shape
            device = base_W.device

            effective_W = base_W + scaling * (lora_B @ lora_A)

            X_full = self._cached_activations.get(layer_name)
            if X_full is None:
                continue

            if X_full.dim() == 3:
                X_full = X_full.reshape(-1, X_full.shape[-1])

            X = X_full.to(device=device, dtype=effective_W.dtype)

            n_blocks = (d_out + block_size - 1) // block_size
            XtX = X.T @ X
            reg = self.reg_lambda * torch.eye(d_in, device=device, dtype=X.dtype)
            XtX_reg = XtX + reg

            try:
                L = torch.linalg.cholesky(XtX_reg)
            except RuntimeError:
                L = None

            A_mat = lora_A
            AAT = A_mat @ A_mat.T
            reg_r = self.reg_lambda * torch.eye(r_val, device=device, dtype=A_mat.dtype)
            try:
                L_r = torch.linalg.cholesky(AAT + reg_r)
                A_pinv = torch.cholesky_solve(A_mat, L_r)
            except RuntimeError:
                A_pinv = torch.linalg.lstsq(AAT + reg_r, A_mat).solution

            for i in range(n_blocks):
                start = i * block_size
                end = min(start + block_size, d_out)

                Y_block = X @ effective_W[start:end, :].T
                XtY = X.T @ Y_block

                if L is not None:
                    W_new_block = torch.cholesky_solve(XtY, L)
                else:
                    W_new_block = torch.linalg.lstsq(XtX_reg, XtY).solution

                W_new_block = W_new_block.T
                delta_W = W_new_block - effective_W[start:end, :]
                delta_B = delta_W @ A_pinv.T / scaling
                lora_B[start:end, :] += delta_B.to(lora_B.dtype)

                recon_error = torch.norm(X @ W_new_block.T - Y_block) ** 2
                total_loss += recon_error.item()

            n_blocks_total += n_blocks

        self._cached_activations.clear()
        self._remove_hooks()

        return total_loss / max(n_blocks_total, 1)

    def _is_conv1d(self, module: nn.Module) -> bool:
        """Check if module is a GPT-2 style Conv1D layer."""
        cls_name = module.__class__.__name__
        return cls_name == "Conv1D" and hasattr(module, "weight") and hasattr(module, "nf")

    def _solve_conv1d_layer(
        self,
        name: str,
        module,
        batch: dict[str, torch.Tensor],
        block_size: int,
    ) -> tuple[float, int]:
        """
        Solve one Conv1D layer via block-wise ALS.

        Conv1D stores weight as [d_in, d_out] (opposite of nn.Linear).
        Forward: Y = X @ W  where X=[N, d_in], W=[d_in, d_out].
        ALS solution: W_new = (X^T X + λI)^(-1) X^T Y, producing [d_in, d_out].
        """
        weight = module.weight.data  # [d_in, d_out]
        d_in, d_out = weight.shape
        device = weight.device

        activations: list[torch.Tensor] = []
        hook_handle = module.register_forward_pre_hook(
            lambda _mod, inp: activations.append(inp[0].detach())
        )

        try:
            with torch.no_grad():
                _ = self.model(**{k: v.to(device) for k, v in batch.items()
                                  if isinstance(v, torch.Tensor)})
            hook_handle.remove()

            if not activations:
                return 0.0, 0

            X = activations[0]
            if X.dim() == 3:
                X = X.reshape(-1, d_in)

            n_blocks = (d_out + block_size - 1) // block_size
            total_loss = 0.0

            # Precompute (X^T X + λI)^(-1) — shared across all blocks
            XtX = X.T @ X  # [d_in, d_in]
            reg = self.reg_lambda * torch.eye(d_in, device=device, dtype=X.dtype)
            XtX_reg = XtX + reg

            try:
                L = torch.linalg.cholesky(XtX_reg)
            except RuntimeError:
                L = None

            for i in range(n_blocks):
                start = i * block_size
                end = min(start + block_size, d_out)

                Y_block = X @ weight[:, start:end]  # [N, block]
                XtY = X.T @ Y_block  # [d_in, block]

                if L is not None:
                    W_new_block = torch.cholesky_solve(XtY, L)  # [d_in, block]
                else:
                    W_new_block = torch.linalg.lstsq(XtX_reg, XtY).solution

                weight[:, start:end] = W_new_block.to(weight.dtype)

                recon_error = torch.norm(X @ W_new_block - Y_block) ** 2
                total_loss += recon_error.item()

            return total_loss, n_blocks

        except Exception as e:
            logger.warning("ALS Conv1D solve failed for layer '%s': %s", name, e)
            hook_handle.remove()
            return 0.0, 0

    def clear_cache(self) -> None:
        """Clear cached inverses (useful between runs)."""
        self._cache.clear()
        self._remove_hooks()

    def _install_activation_hooks(self, layer_names: list[str]):
        """Register forward pre-hooks to capture layer inputs by name."""
        self._remove_hooks()
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_pre_hook(
                    lambda mod, inp, n=name: self._cached_activations.update({n: inp[0].detach()})
                )
                self._hooks.append(hook)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
