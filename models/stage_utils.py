import torch.nn as nn


def split_sequential(seq, num_stages: int):
    """Split an iterable of modules into `num_stages` contiguous nn.Sequential chunks."""
    layers = list(seq)
    if not 1 <= num_stages <= len(layers):
        raise ValueError(f"num_stages must be in [1, {len(layers)}], got {num_stages}")
    base, extra = divmod(len(layers), num_stages)
    stages, i = [], 0
    for s in range(num_stages):
        n = base + (1 if s < extra else 0)
        stages.append(nn.Sequential(*layers[i:i + n]))
        i += n
    return stages
