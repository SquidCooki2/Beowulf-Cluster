import torch.nn as nn


def split_sequential(seq, num_stages: int, layer_counts=None):
    """Split an iterable of modules into `num_stages` contiguous nn.Sequential chunks."""
    layers = list(seq)
    if not 1 <= num_stages <= len(layers):
        raise ValueError(f"num_stages must be in [1, {len(layers)}], got {num_stages}")
    if layer_counts is not None:
        counts = list(layer_counts)
        if len(counts) != num_stages or sum(counts) != len(layers) or any(c < 1 for c in counts):
            raise ValueError(f"Invalid layer_counts {counts} for {len(layers)} layers and {num_stages} stages")
    else:
        base, extra = divmod(len(layers), num_stages)
        counts = [base + (1 if s < extra else 0) for s in range(num_stages)]
    stages, i = [], 0
    for n in counts:
        stages.append(nn.Sequential(*layers[i:i + n]))
        i += n
    return stages
