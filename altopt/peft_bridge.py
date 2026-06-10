"""
Bridge between AltOpt framework and HuggingFace PEFT library.

Enables Protocol C: apply the ALS-SGD-Perturbation alternating optimizer
to PEFT-injected LoRA adapters, rather than our standalone LoRALayer.

Key responsibilities:
  1. Wrap a base model with PEFT LoRA via get_peft_model()
  2. Expose only LoRA adapter parameters (lora_A, lora_B) to AltOpt
  3. Forward pass delegation to the PEFT model
  4. Merge/unload for inference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_PEFT_AVAILABLE = False
try:
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    logger.info("peft not installed; PeftBridge requires `pip install peft`")


@dataclass
class AdapterInfo:
    """Metadata for a single LoRA adapter within a layer."""

    lora_A: nn.Parameter
    lora_B: nn.Parameter
    base_weight: torch.Tensor
    r: int
    scaling: float
    layer_name: str


class PeftBridge:
    """
    Adapts AltOpt to operate on PEFT-injected LoRA adapters.

    Usage:
        bridge = PeftBridge(base_model, peft_config)
        model = bridge.peft_model  # use this for forward/backward
        params = list(bridge.trainable_parameters())  # pass to AltOpt
    """

    def __init__(
        self,
        base_model: nn.Module,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
    ):
        if not _PEFT_AVAILABLE:
            raise ImportError("peft is required for PeftBridge. Install with: pip install peft")

        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        peft_config = PeftLoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.peft_model: PeftModel = get_peft_model(base_model, peft_config)
        self._adapter_map: dict[str, AdapterInfo] = {}
        self._map_adapters()

    def _map_adapters(self):
        for name, module in self.peft_model.named_modules():
            if not hasattr(module, "lora_A"):
                continue

            lora_a = module.lora_A.get("default", None) if isinstance(module.lora_A, dict) else module.lora_A
            lora_b = module.lora_B.get("default", None) if isinstance(module.lora_B, dict) else module.lora_B

            if lora_a is None or lora_b is None:
                continue

            scaling = 1.0
            if hasattr(module, "scaling"):
                s = module.scaling
                scaling = s.get("default", 1.0) if isinstance(s, dict) else s

            r_val = getattr(module, "r", {}).get("default", 8) if isinstance(getattr(module, "r", None), dict) else getattr(module, "r", 8)

            self._adapter_map[name] = AdapterInfo(
                lora_A=lora_a,
                lora_B=lora_b,
                base_weight=module.base_layer.weight,
                r=r_val,
                scaling=scaling,
                layer_name=name,
            )

        n_adapters = len(self._adapter_map)
        n_trainable = sum(p.numel() for p in self.trainable_parameters())
        logger.info("PeftBridge: %d adapter layers, %d trainable params", n_adapters, n_trainable)

    def trainable_parameters(self):
        for info in self._adapter_map.values():
            yield info.lora_A
            yield info.lora_B

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(self, **kwargs) -> dict:
        return self.peft_model(**kwargs)

    def merge_and_unload(self) -> nn.Module:
        return self.peft_model.merge_and_unload()

    def get_adapter_info(self, layer_name: str) -> Optional[AdapterInfo]:
        return self._adapter_map.get(layer_name)

    def all_adapter_info(self) -> dict[str, AdapterInfo]:
        return dict(self._adapter_map)
