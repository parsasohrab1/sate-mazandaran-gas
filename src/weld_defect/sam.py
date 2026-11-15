"""SAM (Sharpness-Aware Minimization) optimizer wrapper.

SAM seeks parameters in neighborhoods with uniformly low loss, leading to
flatter minima and better generalization. Typically +0.5-1 F1 point over
plain AdamW.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization" (ICLR 2021).

Usage:
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=1e-4, rho=0.05)

    # In training loop:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)   # perturb weights

    loss2 = criterion(model(x), y)
    loss2.backward()
    optimizer.second_step(zero_grad=True)  # update with gradient at perturbed point
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization wrapper around a base optimizer."""

    def __init__(
        self,
        params,
        base_optimizer_cls: type[Optimizer] = torch.optim.AdamW,
        rho: float = 0.05,
        **base_optimizer_kwargs,
    ) -> None:
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")
        self.rho = rho
        self.base_optimizer_cls = base_optimizer_cls
        self.base_optimizer = base_optimizer_cls(params, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults
        self.state = {}  # per-parameter state (eps perturbations)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Compute the epsilon perturbation and apply it to parameters."""
        grad_norm = self._grad_norm()
        eps = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = self.rho * p.grad / (grad_norm + 1e-12)
                p.add_(e_w)  # ascend to the perturbed point
                if p not in self.state:
                    self.state[p] = {}
                self.state[p]["eps"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Revert the perturbation and do the actual optimization step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p in self.state and "eps" in self.state[p]:
                    p.sub_(self.state[p]["eps"])  # revert to original position

        self.base_optimizer.step()  # update with gradient at perturbed point

        if zero_grad:
            self.zero_grad()

    def step(self, closure: Callable | None = None) -> None:
        """Standard step — NOT used in SAM. Use first_step + second_step instead."""
        raise RuntimeError("SAM requires first_step() + second_step(). See class docstring.")

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def state_dict(self) -> dict:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "rho": self.rho,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.rho = state_dict["rho"]
