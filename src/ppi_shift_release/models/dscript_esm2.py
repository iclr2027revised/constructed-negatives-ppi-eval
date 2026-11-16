"""D-SCRIPT ESM2-150M adapter: executed architecture and backbone provenance.

This module records the executed D-SCRIPT contact-CNN architecture behind the
``dscript_esm2`` rows in all released tables (E001, E002, E003 512-residue, and
E005 1024-residue).

Backbone provenance (ESM2-150M vs ESM2-650M)
--------------------------------------------
All 42 trained ``dscript_esm2`` checkpoints across every experiment cohort
(9 in E001, 3 in E002, 12 in E003 512-residue, and 18 in E005 1024-residue,
including the 6 Fully Random training units) used the 30-layer, 640-dimensional
``facebook/esm2_t30_150M_UR50D`` per-residue token representations (layer 30),
NOT the 33-layer, 1280-dimensional ``esm2_t33_650M_UR50D`` backbone:

1. Run configs and manifests: ``token_encoder = "esm2_t30_150M_UR50D_layer30"``
   in ``e005_core4_1024.yaml``, ``e005_protocol_freeze.yaml``,
   ``backend_adapter_manifest.final.json``, and all 42 unit ``provenance.json``
   files.
2. Token embedding cache contracts: ``embedding_dim = 640``, ``layer = 30``,
   ``model_name = "facebook/esm2_t30_150M_UR50D"``.
3. Checkpoint tensor dimensions: in every one of the 42 ``.pt`` checkpoints,
   the first linear projection weight ``projection.0.weight`` has shape
   ``(100, 640)`` (64,000 ``float32`` elements = 256,000 bytes), mapping the
   640-dimensional ESM2-150M residue tokens to the 100-dimensional D-SCRIPT
   contact subspace. An ESM2-650M backbone would require shape ``(100, 1280)``
   (512,000 bytes).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

MODEL_ID = "dscript_esm2"
MODEL_VERSION = "ppi_shift.dscript_esm2_150m_contact_cnn.v1"

ESM2_BACKBONE_NAME = "facebook/esm2_t30_150M_UR50D"
ESM2_LAYER = 30
ESM2_EMBEDDING_DIM = 640
PROJECTION_DIM = 100
CONTACT_HIDDEN_DIM = 50
CONTACT_KERNEL_SIZE = 7
POOL_WIDTH = 9
EXPECTED_PROJECTION_WEIGHT_BYTES = PROJECTION_DIM * ESM2_EMBEDDING_DIM * 4  # 256,000


class DScriptContactCNN(nn.Module):
    """Executed D-SCRIPT contact-map CNN over ESM2-150M per-residue tokens."""

    def __init__(
        self,
        input_dim: int = ESM2_EMBEDDING_DIM,
        projection_dim: int = PROJECTION_DIM,
        hidden_dim: int = CONTACT_HIDDEN_DIM,
        kernel_size: int = CONTACT_KERNEL_SIZE,
        pool_width: int = POOL_WIDTH,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim != ESM2_EMBEDDING_DIM:
            raise ValueError(
                f"executed dscript_esm2 requires {ESM2_EMBEDDING_DIM}-dimensional "
                f"ESM2-150M token inputs, got {input_dim}"
            )
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.pool_width = pool_width

        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.contact_pair = nn.Linear(2 * projection_dim, hidden_dim)
        self.contact_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.magnitude = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected (batch, length, {self.input_dim}) token tensor, got {tuple(tokens.shape)}"
            )
        return self.projection(tokens)

    def contact_map(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # Pairwise broadcast [|z_a - z_b|, z_a * z_b] over residue grid (B, L_a, L_b, 2*D)
        diff = torch.abs(z_a.unsqueeze(2) - z_b.unsqueeze(1))
        prod = z_a.unsqueeze(2) * z_b.unsqueeze(1)
        pair = torch.cat([diff, prod], dim=-1)
        hidden = torch.relu(self.contact_pair(pair)).permute(0, 3, 1, 2)
        return self.contact_conv(hidden).squeeze(1)

    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
        z_a = self.project_tokens(tokens_a)
        z_b = self.project_tokens(tokens_b)
        cmap = self.contact_map(z_a, z_b)
        pooled = cmap.amax(dim=(-2, -1))
        logit = self.magnitude * (pooled - self.gamma) + self.bias
        return torch.sigmoid(logit)


def inspect_checkpoint_projection_bytes(checkpoint_path: Path) -> dict[str, Any]:
    """Inspect a PyTorch zip checkpoint's ``projection.0.weight`` byte size."""
    with zipfile.ZipFile(checkpoint_path, "r") as archive:
        candidates = [name for name in archive.namelist() if name.endswith("/data/4")]
        if not candidates:
            raise ValueError(f"Checkpoint archive missing data/4 tensor: {checkpoint_path}")
        n_bytes = archive.getinfo(candidates[0]).file_size
    n_floats = n_bytes // 4
    inferred_dim = n_floats // PROJECTION_DIM
    return {
        "projection_0_weight_bytes": n_bytes,
        "projection_0_weight_float32_elements": n_floats,
        "projection_output_dim": PROJECTION_DIM,
        "inferred_input_embedding_dim": inferred_dim,
        "matches_esm2_150m_640d": n_bytes == EXPECTED_PROJECTION_WEIGHT_BYTES,
    }
