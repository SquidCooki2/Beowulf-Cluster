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
        Calculates layer allocations based on node compute capacity.
        node_gflops: dict of {node_id: gflops_value}
        Returns: dict of {node_id: num_layers}
        """
        total_flops = sum(node_gflops.values())
        allocations = {}
        allocated = 0
        node_ids = list(node_gflops.keys())

        for node_id in node_ids[:-1]:
            ratio = node_gflops[node_id] / total_flops
            layers = max(1, int(round(self.num_layers * ratio)))
            allocations[node_id] = layers
            allocated += layers

        # Allocate remainder to the last node
        allocations[node_ids[-1]] = max(1, self.num_layers - allocated)
        return allocations
