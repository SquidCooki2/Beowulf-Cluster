import time
import torch
import logging

class HeteroPartitioner:
    """
    Benchmarks CPU throughput across heterogeneous nodes and partitions
    model layers proportionally to node compute capacity (FLOPS).
    """
    def __init__(self, num_layers: int = 18):
        self.num_layers = num_layers

    @staticmethod
    def benchmark_local_flops(matrix_dim: int = 2048, iterations: int = 5) -> float:
        """
        Runs synthetic matrix multiplications to measure local node GFLOPS.
        """
        logging.info("Running local node FLOPS benchmark...")
        a = torch.randn(matrix_dim, matrix_dim)
        b = torch.randn(matrix_dim, matrix_dim)
        
        # Warmup
        _ = torch.matmul(a, b)
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch.matmul(a, b)
        duration = time.perf_counter() - start
        
        total_ops = iterations * 2 * (matrix_dim ** 3)
        gflops = (total_ops / duration) / 1e9
        logging.info(f"Local Node Performance: {gflops:.2f} GFLOPS")
        return gflops

    def compute_layer_distribution(self, node_gflops: dict) -> dict:
        """
        Calculates layer allocations based on node compute capacity using the
        largest-remainder method so allocations always sum to self.num_layers.
        node_gflops: dict of {node_id: gflops_value}
        Returns: dict of {node_id: num_layers}
        """
        node_ids = list(node_gflops.keys())
        if not node_ids:
            return {}
        if self.num_layers < len(node_ids):
            raise ValueError(
                f"Cannot distribute {self.num_layers} layers across {len(node_ids)} nodes (need >= 1 layer/node)"
            )

        total_flops = sum(node_gflops.values())
        if total_flops <= 0:
            # Uniform fallback if benchmark readings are non-positive
            base, extra = divmod(self.num_layers, len(node_ids))
            return {nid: base + (1 if i < extra else 0) for i, nid in enumerate(node_ids)}

        # Guarantee at least 1 layer per node, then distribute remaining layers proportionally
        remaining = self.num_layers - len(node_ids)
        exact_extra = {nid: remaining * (node_gflops[nid] / total_flops) for nid in node_ids}
        allocations = {nid: 1 + int(exact_extra[nid]) for nid in node_ids}
        leftover = self.num_layers - sum(allocations.values())

        by_remainder = sorted(node_ids, key=lambda nid: exact_extra[nid] - int(exact_extra[nid]), reverse=True)
        for nid in by_remainder[:leftover]:
            allocations[nid] += 1

        return allocations


def gather_cluster_gflops(stage_id: int, num_stages: int, transceiver, local_gflops: float) -> dict:
    """
    All-gather each stage's measured GFLOPS over the pipeline transceiver
    (forward pass 0 -> N-1, then backward broadcast N-1 -> 0) and reset
    transceiver benchmark counters so startup handshake time is excluded.
    """
    if num_stages == 1:
        return {0: local_gflops}

    if stage_id == 0:
        vec = torch.zeros(num_stages, dtype=torch.float32)
        vec[0] = local_gflops
        transceiver.send_forward(vec)
        vec = transceiver.recv_backward()
    elif stage_id == num_stages - 1:
        vec = transceiver.recv_forward()
        vec[stage_id] = local_gflops
        transceiver.send_backward(vec)
    else:
        vec = transceiver.recv_forward()
        vec[stage_id] = local_gflops
        transceiver.send_forward(vec)
        vec = transceiver.recv_backward()
        transceiver.send_backward(vec)

    transceiver.recv_wait_s = 0.0
    transceiver.send_s = 0.0
    transceiver.bytes_sent = 0
    transceiver.bytes_recv = 0
    return {i: float(vec[i].item()) for i in range(num_stages)}
