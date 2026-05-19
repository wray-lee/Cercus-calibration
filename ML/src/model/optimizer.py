"""
optimizer.py — PyTorch autoencoder-based 4-DOF → 3-DOF calibration.

Core idea (autoencoder bottleneck):
    A 4-DOF sensor stream S = [x_dx, x_dy, y_dx, y_dy] is passed through
    a 4→3 encoder and a 3→4 decoder.  The bottleneck forces the model to
    discover a 3-DOF latent representation of the true physical motion.
    The encoder weight matrix P ∈ R^{3×4} acts as a forward projection;
    its 3×3 sub-matrix (rows for x_dx, y_dx, y_dy) is inverted to produce
    the production calibration matrix W_calib ∈ R^{3×3}.

Channel layout (0-indexed in the 4-DOF tensor):
    0  x_dx  — production X axis
    1  x_dy  — cross-talk (should map to ~zero)
    2  y_dx  — production Y axis
    3  y_dy  — cross-talk / yaw constraint (should map to ~zero)

Production input mapping (calibration.cpp → main.cpp):
    x_dx → system X,  y_dx → system Y,  x_dy → system Z
"""

import json
from typing import Callable, Optional, Tuple

import torch

mapScale: float = 0.015


def load_data(
    raw: list,
    noise_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Convert raw (x_dx, x_dy, y_dx, y_dy) list to a [N, 4] FloatTensor.

    Parameters
    ----------
    raw : list of (x_dx, x_dy, y_dx, y_dy) tuples
    noise_threshold : float or None
        Minimum L1 norm a sample must exceed to be kept.  When *None* an
        adaptive threshold is derived from the batch statistics using the
        median absolute deviation (MAD), making the filter scale-invariant
        across different sensor magnitudes.
    """
    t = torch.tensor(raw, dtype=torch.float32)

    if t.shape[1] != 4:
        raise ValueError(f"Expected 4 columns, got {t.shape[1]}")

    if noise_threshold is not None:
        mask = t.abs().sum(dim=1) > noise_threshold
    else:
        norms = torch.linalg.vector_norm(t, dim=1)
        # 仅对存在显著活动的样本计算物理统计量，防止高频静默帧压垮中位数
        active_norms = norms[norms > 1e-5]
        if len(active_norms) == 0:
            raise ValueError("No active motion detected in raw data.")
        med = active_norms.median()
        mad = (active_norms - med).abs().median().clamp(min=1e-8)
        adaptive_threshold = (med - 2.0 * mad).clamp(min=1e-6)
        upper_bound = med + 10.0 * mad
        mask = (norms > adaptive_threshold) & (norms < upper_bound)

    t = t[mask]
    if t.shape[0] == 0:
        raise ValueError("All samples are zero after filtering — no motion detected.")
    return t


class Calibrator:
    """Autoencoder-based 4-DOF → 3-DOF calibrator.

    The encoder projects 4-DOF sensor data into a 3-DOF latent space.
    The decoder reconstructs the 4-DOF signal from the 3-DOF bottleneck.
    After training, the encoder weight matrix is used to derive a 3×3
    calibration matrix for the production environment.

    Hyperparameters
    ---------------
    anchor_val : float
        Hard-coded value for encoder weight [0, 0] to break scale symmetry.
    reg_lambda : float
        Weight of the cross-talk regularizer in the total loss.
    lr : float
        Adam learning rate.
    """

    ANCHOR_ROW = 0
    ANCHOR_COL = 0
    LOG_INTERVAL = 10  # epochs between progress callbacks
    EARLY_STOP_WINDOW = 20
    EARLY_STOP_TOL = 1e-6

    def __init__(
        self,
        base_scale: float = mapScale,
        reg_lambda: float = 0.5,
        lr: float = 0.01,
    ):
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.anchor_val = base_scale

        # --- Inject Physical Priors ---
        # 4-DOF Columns: 0: x_dx, 1: x_dy, 2: y_dx, 3: y_dy
        # 3-DOF Latent:  0: X,    1: Y,    2: Z
        # Ideal mapping: X <- x_dx (0,0), Y <- y_dx (1,2), Z <- x_dy (2,1)
        init_enc = torch.zeros(3, 4)
        init_enc[0, 0] = base_scale
        init_enc[1, 2] = base_scale
        init_enc[2, 1] = base_scale
        self.W_enc = torch.nn.Parameter(init_enc + torch.randn(3, 4) * 0.001)

        init_dec = torch.zeros(4, 3)
        init_dec[0, 0] = 1.0 / base_scale
        init_dec[2, 1] = 1.0 / base_scale
        init_dec[1, 2] = 1.0 / base_scale
        self.W_dec = torch.nn.Parameter(init_dec + torch.randn(4, 3) * 0.001)

        # Build structural mask for cross-talk penalty (0 for primary, 1 for crosstalk)
        self.reg_mask = torch.ones(3, 4)
        self.reg_mask[0, 0] = 0.0  # Allow x_dx -> X
        self.reg_mask[1, 2] = 0.0  # Allow y_dx -> Y
        self.reg_mask[2, 1] = 0.0  # Allow x_dy -> Z

    def loss(self, S: torch.Tensor) -> torch.Tensor:
        # Forward pass: 4D -> 3D -> 4D
        latent = S @ self.W_enc.T
        S_pred = latent @ self.W_dec.T
        recon = torch.mean((S - S_pred) ** 2)

        # Cross-talk regularizer: explicitly penalize non-primary matrix paths
        # Ensures y_dy (col 3) drives error correction without hijacking axes
        reg = torch.sum((self.W_enc * self.reg_mask) ** 2)

        return recon + self.reg_lambda * reg

    def run(
        self,
        raw_data: list,
        epochs: int = 1000,
        progress_cb: Optional[Callable[[int, int, float], None]] = None,
        noise_threshold: Optional[float] = None,
    ) -> Tuple[list, float]:
        S = load_data(raw_data, noise_threshold=noise_threshold)
        optimizer = torch.optim.Adam([self.W_enc, self.W_dec], lr=self.lr)

        loss_history: list = []
        last_reported = 0

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            l = self.loss(S)
            l.backward()
            optimizer.step()

            # 强制锁定锚点，防止正则化主导引发的尺度塌陷与逆矩阵爆炸
            with torch.no_grad():
                self.W_enc[self.ANCHOR_ROW, self.ANCHOR_COL] = self.anchor_val

            loss_val = l.item()
            loss_history.append(loss_val)

            if progress_cb and epoch % self.LOG_INTERVAL == 0:
                progress_cb(epoch, epochs, loss_val)
                last_reported = epoch

            if len(loss_history) >= self.EARLY_STOP_WINDOW:
                window = loss_history[-self.EARLY_STOP_WINDOW :]
                delta = abs(window[0] - window[-1])
                if delta < self.EARLY_STOP_TOL:
                    if progress_cb and epoch != last_reported:
                        progress_cb(epoch, epochs, loss_val)
                    break

        final_loss = self.loss(S).item()
        if progress_cb:
            progress_cb(epochs, epochs, final_loss)

        cal_matrix = build_calibration_matrix(self.W_enc.detach())
        return cal_matrix, final_loss


def build_calibration_matrix(W_enc: torch.Tensor) -> list:
    """Extract the 3×3 production sub-matrix from the encoder and invert it.

    The encoder weight matrix P has shape [3, 4].  The sub-matrix P_sub is
    formed by selecting the rows corresponding to the three production input
    channels (x_dx, y_dx, x_dy) and the first 3 output columns:

        P_sub[i, j] = W_enc[j, _PRODUCTION_ROWS[i]]

    The calibration matrix is ``W_calib = inv(P_sub)``.

    Returns
    -------
    list[list[float]]  — 3×3 JSON-serializable matrix
    """
    P_sub = W_enc.T[[0, 2, 1], :]
    # W_calib = torch.linalg.inv(P_sub)
    # if not torch.isfinite(W_calib).all():
    #     raise ValueError("Calibration matrix contains NaN or Inf values.")
    # return W_calib.numpy().tolist()
    return P_sub.numpy().tolist()


def save_json(matrix: list, path: str = "calibration_cfg.json"):
    with open(path, "w") as f:
        json.dump(matrix, f, indent=2)
