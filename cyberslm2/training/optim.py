"""
optim.py
========
Muon (MomentUm Orthogonalized by Newton-schulz) with an AdamW fallback.

Idea
----
SGD-momentum produces an update matrix whose singular values are wildly
unbalanced: a few directions dominate and the rest barely move. Muon replaces
the update G with its nearest semi-orthogonal matrix (all singular values ~1),
so every direction in the weight matrix receives a comparable amount of update.
In practice this reaches a target loss in noticeably fewer steps than AdamW at
the same batch size, and it needs one momentum buffer instead of two moments,
so it also costs less memory.

The orthogonalization uses a quintic Newton-Schulz iteration rather than an SVD:

    X <- a*X + (b*A + c*A^2) X,   A = X X^T

with (a, b, c) = (3.4445, -4.775, 2.0315), tuned so the iteration converges fast
from a Frobenius-normalized start. Five steps suffice. It runs in bf16 because
only the *direction* of the result matters.

Scope
-----
Muon applies to 2-D hidden weight matrices only. Embeddings, the LM head, norm
gains and any 1-D tensor keep AdamW: their geometry is not the matrix-valued one
Muon assumes, and orthogonalizing an embedding table is meaningless.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch import Tensor


@torch.no_grad()
def zeropower_via_newtonschulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Approximate the orthogonal polar factor of G (i.e. U V^T of its SVD).

    Operates on the transposed matrix when G is tall so the Gram matrix A is
    always the smaller of the two possible products.
    """
    if G.ndim != 2:
        raise ValueError(f"Newton-Schulz expects a 2-D matrix, got shape {tuple(G.shape)}")

    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)          # convergence needs ||X|| <= 1

    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon for 2-D parameters, with a built-in AdamW group for everything else.

    Parameter groups carry a ``use_muon`` flag; build them with
    :func:`build_optimizer` rather than by hand.
    """

    def __init__(
        self,
        param_groups: Iterable[dict[str, Any]],
        lr: float = 3e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            weight_decay=weight_decay,
            use_muon=True,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    # -- update rules -------------------------------------------------------

    def _muon_step(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            update = g.add(buf, alpha=momentum) if group["nesterov"] else buf

            ortho = zeropower_via_newtonschulz(update, steps=group["ns_steps"])

            # A wide matrix needs a larger step than a square one to move its
            # output by the same amount; this keeps the effective step size
            # consistent across differently shaped layers.
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(ortho, alpha=-lr * scale)

    def _adamw_step(self, group: dict[str, Any]) -> None:
        lr = group.get("adamw_lr", group["lr"])
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            t = state["step"]
            m, v = state["exp_avg"], state["exp_avg_sq"]

            m.mul_(beta1).add_(g, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            bias1 = 1 - beta1 ** t
            bias2 = 1 - beta2 ** t
            denom = (v.sqrt() / (bias2 ** 0.5)).add_(eps)

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.addcdiv_(m / bias1, denom, value=-lr)


def build_optimizer(model, cfg) -> torch.optim.Optimizer:
    """
    Split parameters into Muon / AdamW-decay / AdamW-no-decay groups.

    Rules:
      * 2-D weights inside blocks  -> Muon
      * embedding and LM head      -> AdamW, no weight decay
      * norm gains and 1-D tensors -> AdamW, no weight decay

    Weight decay on an embedding pulls rare-token vectors toward zero purely
    because they are rarely updated, which is the opposite of what you want.
    """
    muon_params, adamw_decay, adamw_no_decay = [], [], []
    seen: set[int] = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))

        is_embedding = "embed_tokens" in name or "lm_head" in name
        is_norm_or_1d = p.ndim < 2

        if is_embedding or is_norm_or_1d:
            adamw_no_decay.append(p)
        elif p.ndim == 2 and cfg.optimizer == "muon":
            muon_params.append(p)
        else:
            adamw_decay.append(p)

    groups = []
    if muon_params:
        groups.append({"params": muon_params, "use_muon": True,
                       "weight_decay": cfg.weight_decay})
    if adamw_decay:
        groups.append({"params": adamw_decay, "use_muon": False,
                       "weight_decay": cfg.weight_decay})
    if adamw_no_decay:
        groups.append({"params": adamw_no_decay, "use_muon": False,
                       "weight_decay": 0.0})

    if cfg.optimizer == "muon":
        return Muon(
            groups,
            lr=cfg.lr,
            momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps,
            adamw_lr=cfg.adamw_lr,
            adamw_betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

    for g in groups:
        g["use_muon"] = False
    return torch.optim.AdamW(
        [{"params": g["params"], "weight_decay": g["weight_decay"]} for g in groups],
        lr=cfg.adamw_lr,
        betas=(cfg.beta1, cfg.beta2),
    )
