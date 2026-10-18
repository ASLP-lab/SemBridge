"""Load the released SemBridge generator."""

from pathlib import Path

import torch
from safetensors.torch import load_file

from .models.configuration import GeneratorConfig
from .models.generator import SemBridgeGenerator


def load_generator_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SemBridgeGenerator:
    """Load config.json and model.safetensors from a model directory."""

    root = Path(checkpoint)
    config = GeneratorConfig.model_validate_json(
        (root / "config.json").read_text(encoding="utf-8")
    )
    model = SemBridgeGenerator(config).to(device=device, dtype=dtype)
    model.load_state_dict(load_file(str(root / "model.safetensors"), device="cpu"))
    return model.eval()
