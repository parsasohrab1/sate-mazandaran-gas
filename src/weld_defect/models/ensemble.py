"""Ensemble model — averages softmax across member models.

Implements ADR-1 (Architecture.md): diversity of inductive bias
(texture + geometry + light baseline) reduces correlated errors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    """Ensemble of member models with softmax averaging."""

    def __init__(self, members: list[nn.Module], strategy: str = "softmax_avg") -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.strategy = strategy
        if strategy not in ("softmax_avg", "voting", "stacking"):
            raise ValueError(f"Unknown ensemble strategy: {strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.strategy == "softmax_avg":
            probs = None
            for m in self.members:
                logits = m(x)
                p = F.softmax(logits, dim=1)
                probs = p if probs is None else probs + p
            probs = probs / len(self.members)
            # Return log-probs as logits for compatibility with CrossEntropy/FocalLoss.
            return torch.log(probs + 1e-8)

        if self.strategy == "voting":
            # Hard voting — returns one-hot of majority (as logits).
            preds = torch.stack([m(x).argmax(dim=1) for m in self.members], dim=1)
            # Majority vote per sample.
            voted = torch.mode(preds, dim=1).values
            logits = torch.full(
                (x.shape[0], self.members[0](x).shape[1]),
                -1e8,
                device=x.device,
                dtype=torch.float32,
            )
            logits[torch.arange(x.shape[0]), voted] = 0.0
            return logits

        if self.strategy == "stacking":
            # Concatenate member logits; a learned meta-classifier would go here.
            logits = torch.cat([m(x) for m in self.members], dim=1)
            return logits

        raise RuntimeError("unreachable")
