"""
GradientStabilizer: per-parameter gradient magnitude stabilization for
adaptive optimizers via EMA-based normalization.

For each parameter, the wrapper rescales the gradient so that its norm
matches a stable target:

    target = E[||g||] / sqrt(E[||g||^2] )+ eps

where the expectations are running EMAs with coefficients gamma1 and
gamma2. The result lies in (0, 1]: stable directions stay close to unit
norm, noisy directions are damped in proportion to their coefficient of
variation. Designed to sit in front of Adam / AdamW and similar adaptive
methods.

Usage:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = GSWrapper(optimizer, gamma1=0.6, gamma2=0.999)
"""

from typing import Dict, Iterable, Iterator, Union

import torch
from torch.optim import Optimizer


class GradientStabilizer:
    """EMA-based gradient magnitude stabilizer (single-parameter logic).

    Args:
        gamma1: EMA coefficient for E[||g||]. Larger -> longer memory.
        gamma2: EMA coefficient for E[||g||^2]. Typically close to 1.
        eps: numerical floor used in divisions and sqrt.
        bias_correction: apply Adam-style ``1 / (1 - gamma**step)``
            correction. ``step`` is the count of *successful* EMA updates
            (skipped iterations do not advance it), so the correction
            stays consistent with the actual EMA history.
    """

    def __init__(
        self,
        gamma1: float = 0.6,
        gamma2: float = 0.999,
        eps: float = 1e-12,
        bias_correction: bool = True,
    ):
        if not 0.0 <= gamma1 < 1.0:
            raise ValueError(f"gamma1 must be in [0, 1), got {gamma1}")
        if not 0.0 <= gamma2 < 1.0:
            raise ValueError(f"gamma2 must be in [0, 1), got {gamma2}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.eps = float(eps)
        self.bias_correction = bool(bias_correction)

        # Per-parameter state, keyed by id(p) at runtime.
        # GSWrapper translates id -> stable positional index for
        # serialization, so checkpoints survive a process restart.
        self.state: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop all accumulated EMA state."""
        self.state.clear()

    def _get_state(
        self,
        key: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        st = self.state.get(key)
        if st is None:
            st = {
                "mnorm": torch.zeros((), device=device, dtype=dtype),
                "vnorm": torch.zeros((), device=device, dtype=dtype),
                "step": 0,     # number of successful EMA updates
                "skipped": 0,  # diagnostic: non-finite / zero-norm events
            }
            self.state[key] = st
        return st

    # ------------------------------------------------------------------
    # EMA math
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_linear_(
        accum: torch.Tensor, new: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """In-place: accum <- gamma * accum + (1 - gamma) * new."""
        accum.mul_(gamma).add_(new, alpha=(1.0 - gamma))
        return accum

    def _maybe_bias_correct(
        self, val: torch.Tensor, gamma: float, step: int
    ) -> torch.Tensor:
        """Standard Adam-style bias correction, applied only when enabled.

        Returns a NEW tensor (does not alias the stored EMA), so callers
        may safely apply in-place ops to the result.
        """
        if not self.bias_correction:
            return val
        bc = 1.0 - (gamma ** step)
        return val / bc

    # ------------------------------------------------------------------
    # Core scaling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _scale_param(self, p: torch.nn.Parameter) -> None:
        if p.grad is None:
            return
        g = p.grad
        if g.is_sparse:
            return  # sparse grads not supported

        key = id(p)
        st = self._get_state(key, device=g.device)

        # 1) Compute gradient norm in fp32 BEFORE touching any state.
        #    Operating on `g` directly (not on a flattened view) keeps
        #    non-contiguous gradients correct: an earlier version used
        #    `g.contiguous().view(-1)` which returns a copy whose
        #    in-place mutations never reach `p.grad`.
        g_norm = g.norm(p=2).to(torch.float32)

        if not torch.isfinite(g_norm) or g_norm <= 0:
            # Skip cleanly: no step bump, no EMA update.
            st["skipped"] += 1
            return

        # 2) Advance the EMA-update counter only on real updates so that
        #    bias correction's `1 - gamma**step` matches the number of
        #    samples actually seen by the EMA.
        st["step"] += 1
        step = st["step"]

        # 3) EMA updates and bias correction share the same counter.
        self._ema_linear_(st["mnorm"], g_norm, self.gamma1)
        m_est = self._maybe_bias_correct(st["mnorm"], self.gamma1, step)

        g_sq = g_norm * g_norm
        self._ema_linear_(st["vnorm"], g_sq, self.gamma2)
        v_est = self._maybe_bias_correct(st["vnorm"], self.gamma2, step)

        # target_over_sigma = E[||g||] / (sqrt(E[||g||^2] )+eps)  in (0, 1]
        denom = v_est.sqrt().add_(self.eps)
        target_over_sigma = (m_est / denom).to(g.dtype)

        # 4) Rescale the original gradient in place. Clamp the norm in
        #    fp32 first so `eps` is not lost when casting to low-precision
        #    dtypes (e.g. fp16, where 1e-12 underflows to zero).
        denom_g = g_norm.clamp(min=self.eps).to(g.dtype)
        g.mul_(target_over_sigma / denom_g)

    @torch.no_grad()
    def __call__(
        self,
        obj: Union[Optimizer, Iterable, list, tuple],
    ) -> None:
        """Apply scaling to all parameters in ``obj``.

        ``obj`` may be an :class:`Optimizer`, a list of param_group
        dicts, or any iterable of parameters.
        """
        if isinstance(obj, Optimizer):
            for group in obj.param_groups:
                for p in group["params"]:
                    self._scale_param(p)
            return

        if (
            isinstance(obj, (list, tuple))
            and len(obj) > 0
            and isinstance(obj[0], dict)
        ):
            for group in obj:
                for p in group["params"]:
                    self._scale_param(p)
            return

        for p in obj:
            self._scale_param(p)


class GSWrapper:
    """Optimizer wrapper that rescales gradients via
    :class:`GradientStabilizer` before each ``step()``.

    Forwards ``param_groups``, ``state``, ``defaults``, ``zero_grad``,
    ``step``, and ``add_param_group`` to the wrapped optimizer.
    ``state_dict`` / ``load_state_dict`` persist both the optimizer
    state and the stabilizer's per-parameter EMAs, keyed by parameter
    position (not Python ``id``), so checkpoints survive a process
    restart.

    Example:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt = GSWrapper(opt, gamma1=0.6, gamma2=0.999)
    """

    def __init__(self, optimizer: Optimizer, **scaler_kwargs):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"GSWrapper expects a torch.optim.Optimizer, "
                f"got {type(optimizer).__name__}"
            )
        self.optimizer = optimizer
        self.scaler = GradientStabilizer(**scaler_kwargs)

    # ------------------------------------------------------------------
    # Pass-through properties / methods
    # ------------------------------------------------------------------

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    @property
    def defaults(self):
        return self.optimizer.defaults

    def add_param_group(self, param_group: dict) -> None:
        self.optimizer.add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # Rescale gradients, then delegate to the base optimizer.
        self.scaler(self.optimizer)
        if closure is not None:
            return self.optimizer.step(closure)
        return self.optimizer.step()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _params_in_order(self) -> Iterator[torch.nn.Parameter]:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                yield p

    def state_dict(self) -> dict:
        """Returns a checkpoint-safe state dict.

        Scaler state is re-keyed from runtime ``id(p)`` to positional
        index so that loading into a fresh process (where parameter
        ``id``s differ) restores the correct EMAs.
        """
        scaler_state_by_idx = {}
        for i, p in enumerate(self._params_in_order()):
            st = self.scaler.state.get(id(p))
            if st is None:
                continue
            scaler_state_by_idx[i] = {
                "mnorm": st["mnorm"].detach().cpu().clone(),
                "vnorm": st["vnorm"].detach().cpu().clone(),
                "step": int(st["step"]),
                "skipped": int(st.get("skipped", 0)),
            }
        return {
            "optimizer": self.optimizer.state_dict(),
            "scaler_state_by_idx": scaler_state_by_idx,
            "scaler_hparams": {
                "gamma1": self.scaler.gamma1,
                "gamma2": self.scaler.gamma2,
                "eps": self.scaler.eps,
                "bias_correction": self.scaler.bias_correction,
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])

        # Warn (but do not crash) on hyperparameter mismatch.
        hp = state_dict.get("scaler_hparams")
        if hp is not None:
            for k, v in hp.items():
                cur = getattr(self.scaler, k, None)
                if cur is not None and cur != v:
                    print(
                        f"[GSWrapper] scaler hyperparameter '{k}' "
                        f"differs from checkpoint: ckpt={v}, current={cur}"
                    )

        scaler_by_idx = state_dict.get("scaler_state_by_idx", {})
        self.scaler.state.clear()
        for i, p in enumerate(self._params_in_order()):
            blob = scaler_by_idx.get(i)
            if blob is None:
                continue
            dev = p.device
            self.scaler.state[id(p)] = {
                "mnorm": blob["mnorm"].to(dev),
                "vnorm": blob["vnorm"].to(dev),
                "step": int(blob["step"]),
                "skipped": int(blob.get("skipped", 0)),
            }
