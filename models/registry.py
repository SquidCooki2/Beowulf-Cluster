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


def _malaria(num_stages=5, pretrained=True, **kw):
    from models.malaria_mobilenet import create_malaria_stages
    return create_malaria_stages(num_stages, pretrained=pretrained)


REGISTRY = {
    "malaria_mobilenet": ModelSpec("malaria_mobilenet", _malaria, (3, 128, 128), 2, "ce", True, "malaria"),
}
DEFAULT_MODEL = "malaria_mobilenet"


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
