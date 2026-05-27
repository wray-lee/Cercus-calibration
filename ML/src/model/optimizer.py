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
    0  x_dx  — physical X (roll)    → UI X
    1  x_dy  — physical Z (yaw)     → UI Z
    2  y_dx  — physical Y (pitch)   → UI Y
    3  y_dy  — cross-talk / unused

Production input mapping (calibration.cpp → main.cpp):
    x_dx → system X,  y_dx → system Y,  x_dy → system Z
"""

import json
from typing import Callable, Optional, Tuple
import math

import torch

mapScale: float = 60 * math.pi / 4000


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
        Hard-coded value for encoder weight [0, 2] to break scale symmetry.
    reg_lambda : float
        Weight of the correlation regularizer in the total loss.
    lr : float
        Adam learning rate.
    """

    # Anchor Latent X (Row 0) to Sensor Y's dx (Col 2)
    ANCHOR_ROW = 0
    ANCHOR_COL = 2
    LOG_INTERVAL = 10  # epochs between progress callbacks
    EARLY_STOP_WINDOW = 20
    EARLY_STOP_TOL = 1e-6

    def __init__(
        self,
        base_scale: float = mapScale,
        reg_lambda: float = 0.1,  # Balanced penalty for L2
        lr: float = 0.01,
    ):
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.anchor_val = base_scale

        # --- Strict Diagonal Physical Priors ---
        init_enc = torch.zeros(3, 4)
        init_enc[0, 0] = base_scale  # raw_dx (S col 0) -> UI X
        init_enc[1, 2] = base_scale  # raw_dy (S col 2) -> UI Y
        init_enc[2, 1] = base_scale  # raw_dz (S col 1) -> UI Z
        self.W_enc = torch.nn.Parameter(init_enc + torch.randn(3, 4) * 0.0001)

        init_dec = torch.zeros(4, 3)
        init_dec[0, 0] = 1.0 / base_scale
        init_dec[2, 1] = 1.0 / base_scale
        init_dec[1, 2] = 1.0 / base_scale
        self.W_dec = torch.nn.Parameter(init_dec + torch.randn(4, 3) * 0.0001)

        # L2 Penalty Mask
        self.reg_mask = torch.ones(3, 4)
        self.reg_mask[0, 0] = 0.0  # Main UI X
        self.reg_mask[1, 2] = 0.0  # Main UI Y
        self.reg_mask[2, 1] = 0.0  # Main UI Z

    def loss(self, S: torch.Tensor) -> torch.Tensor:
        latent = S @ self.W_enc.T
        S_pred = latent @ self.W_dec.T

        # 1. Scale-invariant reconstruction loss
        recon = torch.mean((S - S_pred) ** 2) / (torch.var(S) + 1e-6)

        # 2. L2 Ridge Penalty on Cross-talk
        # Punish the absolute magnitude of cross-talk weights relative to base_scale
        # This prevents noise amplification while allowing soft geometric corrections.
        l2_penalty = torch.sum(((self.W_enc * self.reg_mask) / self.anchor_val) ** 2)

        return recon + self.reg_lambda * l2_penalty

    def run(
        self,
        raw_data: list,
        epochs: int = 10000,
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

            # 强制锁定三个主物理轴，阻止绝对尺度爆炸
            with torch.no_grad():
                self.W_enc[0, 0] = self.anchor_val  # raw_dx -> UI X
                self.W_enc[1, 2] = self.anchor_val  # raw_dy -> UI Y
                self.W_enc[2, 1] = self.anchor_val  # raw_dz -> UI Z

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
    """Derive the 3x3 production calibration matrix.

    The autoencoder latent space (Outputs) is: [X_real, Y_real, Z_real]
    The autoencoder input space (S) is: [x_dx (0), x_dy (1), y_dx (2), y_dy (3)]

    In the production engine (hardware.py), the math is:
    Out = M * [raw_dx, raw_dy, raw_dz]^T

    Assuming the physical firmware mapping is:
    raw_dx == x_dx (S col 0)
    raw_dy == y_dx (S col 2)
    raw_dz == x_dy (S col 1)

    We must build a 3x3 matrix M where rows are outputs and columns match
    the [0, 2, 1] input sequence.
    """
    # Extract the required columns for [raw_dx, raw_dy, raw_dz]
    M = W_enc[:, [0, 2, 1]]

    return M.detach().numpy().tolist()


def save_json(matrix: list, path: str = "calibration_cfg.json"):
    with open(path, "w") as f:
        json.dump(matrix, f, indent=2)
