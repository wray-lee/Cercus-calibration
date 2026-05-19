"""
optimizer.py — PyTorch unsupervised 3×3 matrix calibration.

Core idea (dual-point kinematic redundancy):
    Two 2-DOF sensors (S_a, S_b) observe the same physical motion.
    Learn weight matrices W_a, W_b ∈ R^{3×2} such that
        Ω_a = S_a @ W_a^T  ≈  Ω_b = S_b @ W_b^T
    The MSE alignment loss drives both to a shared 3-DOF frame.
    An anchor on W_a[0,0] = 1.0 and a norm regularizer prevent
    the trivial all-zero collapse.
"""

import json
from typing import Callable, Optional, Tuple

import torch


def load_data(
    raw: list,
    noise_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert raw (dx1,dy1,dx2,dy2) list to two [N,2] FloatTensors.

    Parameters
    ----------
    raw : list of (dx1,dy1,dx2,dy2) tuples
    noise_threshold : float or None
        Minimum L2 norm a sample must exceed to be kept.  When *None* an
        adaptive threshold is derived from the batch statistics using the
        median absolute deviation (MAD), making the filter scale-invariant
        across different sensor magnitudes.
    """
    t = torch.tensor(raw, dtype=torch.float32)

    if noise_threshold is not None:
        mask = t.abs().sum(dim=1) > noise_threshold
    else:
        norms = torch.linalg.vector_norm(t, dim=1)
        med = norms.median()
        mad = (norms - med).abs().median().clamp(min=1e-8)
        adaptive_threshold = (med - 2.0 * mad).clamp(min=1e-6)
        mask = norms > adaptive_threshold

    t = t[mask]
    if t.shape[0] == 0:
        raise ValueError("All samples are zero after filtering — no motion detected.")
    return t[:, :2], t[:, 2:]


class Calibrator:
    """Unsupervised dual-sensor calibrator.

    Hyperparameters
    ---------------
    anchor_val : float
        Hard-coded value for W_a[0,0] to break scale symmetry.
    reg_lambda : float
        Weight of the norm regularizer in the total loss.
    lr : float
        Adam learning rate.
    """

    ANCHOR_ROW = 0
    ANCHOR_COL = 0
    LOG_INTERVAL = 10  # epochs between progress callbacks

    def __init__(
        self,
        anchor_val: float = 1.0,
        reg_lambda: float = 0.1,
        lr: float = 0.01,
    ):
        self.anchor_val = anchor_val
        self.reg_lambda = reg_lambda
        self.lr = lr

        self.W_a = torch.randn(3, 2, requires_grad=True)
        self.W_b = torch.randn(3, 2, requires_grad=True)

    def _anchor(self):
        with torch.no_grad():
            self.W_a[self.ANCHOR_ROW, self.ANCHOR_COL] = self.anchor_val

    def loss(self, S_a: torch.Tensor, S_b: torch.Tensor) -> torch.Tensor:
        omega_a = S_a @ self.W_a.T
        omega_b = S_b @ self.W_b.T
        mse = torch.mean((omega_a - omega_b) ** 2)
        norm = (self.W_a ** 2).sum(dim=1).clamp(min=1e-8).sqrt()
        penalty = torch.abs(1.0 - torch.mean(norm))
        return mse + self.reg_lambda * penalty

    EARLY_STOP_WINDOW = 20
    EARLY_STOP_TOL = 1e-6

    def run(
        self,
        raw_data: list,
        epochs: int = 1000,
        progress_cb: Optional[Callable[[int, int, float], None]] = None,
        noise_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Run full training pipeline.

        Parameters
        ----------
        raw_data : list of (dx1,dy1,dx2,dy2) tuples
        epochs : int
        progress_cb : callable(epoch, total_epochs, loss_val) or None
        noise_threshold : float or None
            Forwarded to ``load_data``; see its docstring.

        Returns
        -------
        W_a : Tensor [3,2]
        final_loss : float
        """
        S_a, S_b = load_data(raw_data, noise_threshold=noise_threshold)
        optimizer = torch.optim.Adam([self.W_a, self.W_b], lr=self.lr)

        with torch.no_grad():
            self.W_a[self.ANCHOR_ROW, self.ANCHOR_COL] = self.anchor_val

        loss_history: list = []
        last_reported = 0

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            l = self.loss(S_a, S_b)
            l.backward()
            with torch.no_grad():
                self.W_a.grad[self.ANCHOR_ROW, self.ANCHOR_COL] = 0.0
            optimizer.step()

            loss_val = l.item()
            loss_history.append(loss_val)

            if progress_cb and epoch % self.LOG_INTERVAL == 0:
                progress_cb(epoch, epochs, loss_val)
                last_reported = epoch

            # early stopping: plateau detection over sliding window
            if len(loss_history) >= self.EARLY_STOP_WINDOW:
                window = loss_history[-self.EARLY_STOP_WINDOW:]
                delta = abs(window[0] - window[-1])
                if delta < self.EARLY_STOP_TOL:
                    if progress_cb and epoch != last_reported:
                        progress_cb(epoch, epochs, loss_val)
                    break

        final_loss = self.loss(S_a, S_b).item()
        if progress_cb:
            progress_cb(epochs, epochs, final_loss)
        return self.W_a.detach(), final_loss


def build_matrix_3x3(W_a: torch.Tensor) -> list:
    """Convert a [3,2] weight tensor to a 3×3 JSON-serializable list.

    Column 3 (Z-axis) is set to [0, 0, 1] as a passthrough.
    """
    W = W_a.numpy().tolist()
    return [
        [W[0][0], W[0][1], 0.0],
        [W[1][0], W[1][1], 0.0],
        [W[2][0], W[2][1], 1.0],
    ]


def save_json(matrix: list, path: str = "calibration_cfg.json"):
    with open(path, "w") as f:
        json.dump(matrix, f, indent=2)
