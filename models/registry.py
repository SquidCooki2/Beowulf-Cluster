"""Model registry: one definition, shared by the pipeline engine and HPC training."""
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    create_stages: Callable      # (num_stages, **kw) -> list[nn.Module]
    input_shape: Tuple[int, ...]  # per-sample shape
    num_outputs: int
    loss: str                    # "ce" (classification) or "mse" (regression)
    two_phase: bool              # pretrained backbone: head first, then fine-tune
    data: str                    # module name inside `data/`
    num_layers: int              # partitionable layer/block count


def _malaria(num_stages=5, pretrained=True, layer_counts=None, **kw):
    from models.malaria_mobilenet import create_malaria_stages
    return create_malaria_stages(num_stages, pretrained=pretrained, layer_counts=layer_counts)


def _smolvlm(num_stages=5, pretrained=True, layer_counts=None, **kw):
    from models.scienceqa_smolvlm import create_smolvlm_stages
    return create_smolvlm_stages(num_stages, pretrained=pretrained, layer_counts=layer_counts)


REGISTRY = {
    "malaria_mobilenet": ModelSpec("malaria_mobilenet", _malaria, (3, 128, 128), 2, "ce", True, "malaria", 19),
    "scienceqa_smolvlm": ModelSpec("scienceqa_smolvlm", _smolvlm, (4, 128, 128), 5, "ce", True, "scienceqa", 30),
}
DEFAULT_MODEL = "malaria_mobilenet"


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
