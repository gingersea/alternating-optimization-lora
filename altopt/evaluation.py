"""
Unified evaluation protocol for fair comparison across all four protocols.

The Evaluator is a stateless module that guarantees identical evaluation
conditions — same data, same batch size, same metric implementation —
regardless of which optimizer or parameter form is being tested.

This is the "统一评分" (unified scoring) component that addresses the
fundamental confound: without identical evaluation, any observed difference
between protocols cannot be attributed to the variables of interest.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Stateless evaluation runner with standardized metric computation.

    Usage:
        evaluator = Evaluator(["perplexity", "loss"], eval_dataset)
        results = evaluator.evaluate(peft_bridge.peft_model)
        # results = {"perplexity": 18.3, "loss": 2.91, "n_tokens": 12345}
    """

    def __init__(
        self,
        metrics: list[str],
        eval_dataloader,
        batch_size: int = 8,
    ):
        self.metrics = metrics
        self.eval_dataloader = eval_dataloader
        self.batch_size = batch_size
        self._downstream_tasks: dict[str, dict] = {}

    def evaluate(self, model: nn.Module) -> dict[str, float]:
        model.eval()
        device = next(model.parameters()).device

        accumulators: dict[str, float] = {"n_tokens": 0}
        for metric in self.metrics:
            accumulators[metric] = 0.0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                n_tokens = self._count_tokens(batch)
                accumulators["n_tokens"] += n_tokens
                accumulators["loss"] += loss.item() * n_tokens

                if "perplexity" in self.metrics:
                    pass  # computed from aggregated loss

        results: dict[str, float] = {}
        total_tokens = max(accumulators["n_tokens"], 1)

        if "loss" in self.metrics:
            results["loss"] = accumulators["loss"] / total_tokens

        if "perplexity" in self.metrics:
            avg_loss = results.get("loss", accumulators["loss"] / total_tokens)
            results["perplexity"] = torch.exp(torch.tensor(avg_loss)).item()

        results["n_tokens"] = total_tokens

        model.train()
        return results

    def _count_tokens(self, batch: dict[str, torch.Tensor]) -> int:
        if "input_ids" in batch:
            mask = batch.get("attention_mask", None)
            if mask is not None:
                return mask.sum().item()
            return batch["input_ids"].numel()
        # Generic fallback: count elements in the first tensor
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.numel()
        return 1

    def add_downstream_task(self, name: str, dataset, metric_fn: Callable):
        self._downstream_tasks[name] = {"dataset": dataset, "metric_fn": metric_fn}
