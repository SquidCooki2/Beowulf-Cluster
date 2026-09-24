import zmq
import torch
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

class TensorTransceiver:
    """
    Non-blocking point-to-point tensor transport over ZeroMQ sockets.
    Handles serialization/deserialization of PyTorch tensors across cluster nodes.
    """
    def __init__(self, stage_id: int, config: dict):
        self.stage_id = stage_id
        self.config = config
        self.stages = config["stages"]
        self.num_stages = len(self.stages)
        self.context = zmq.Context()

        # 1. Receiver Sockets (Bind on local node)
        if self.stage_id > 0:
            # Receive forward activations from (stage_id - 1)
            self.recv_fwd_sock = self.context.socket(zmq.PULL)
            fwd_port = self.stages[stage_id]["fwd_port"]
            self.recv_fwd_sock.bind(f"tcp://0.0.0.0:{fwd_port}")
            logging.info(f"Stage {stage_id}: Bound Forward PULL socket on port {fwd_port}")

        if self.stage_id < self.num_stages - 1:
            # Receive backward gradients from (stage_id + 1)
            self.recv_bwd_sock = self.context.socket(zmq.PULL)
            bwd_port = self.stages[stage_id]["bwd_port"]
            self.recv_bwd_sock.bind(f"tcp://0.0.0.0:{bwd_port}")
            logging.info(f"Stage {stage_id}: Bound Backward PULL socket on port {bwd_port}")

        # 2. Sender Sockets (Connect to neighboring nodes)
        if self.stage_id < self.num_stages - 1:
            # Push forward activations to (stage_id + 1)
            next_ip = self.stages[stage_id + 1]["ip"]
            next_fwd_port = self.stages[stage_id + 1]["fwd_port"]
            self.send_fwd_sock = self.context.socket(zmq.PUSH)
            self.send_fwd_sock.connect(f"tcp://{next_ip}:{next_fwd_port}")
            logging.info(f"Stage {stage_id}: Connected Forward PUSH socket to {next_ip}:{next_fwd_port}")

        if self.stage_id > 0:
            # Push backward gradients to (stage_id - 1)
            prev_ip = self.stages[stage_id - 1]["ip"]
            prev_bwd_port = self.stages[stage_id - 1]["bwd_port"]
            self.send_bwd_sock = self.context.socket(zmq.PUSH)
            self.send_bwd_sock.connect(f"tcp://{prev_ip}:{prev_bwd_port}")
            logging.info(f"Stage {stage_id}: Connected Backward PUSH socket to {prev_ip}:{prev_bwd_port}")

    def _serialize(self, tensor: torch.Tensor):
        meta = json.dumps({
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", "")
        }).encode("utf-8")
        data = tensor.detach().cpu().contiguous().numpy().tobytes()
        return [meta, data]

    def _deserialize(self, meta_bytes: bytes, data_bytes: bytes) -> torch.Tensor:
        meta = json.loads(meta_bytes.decode("utf-8"))
        dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "int64": np.int64,
            "int32": np.int32
        }
        np_dtype = dtype_map.get(meta["dtype"], np.float32)
        np_arr = np.frombuffer(data_bytes, dtype=np_dtype).reshape(meta["shape"]).copy()
        return torch.from_numpy(np_arr)

    def send_forward(self, tensor: torch.Tensor):
        self.send_fwd_sock.send_multipart(self._serialize(tensor))

    def recv_forward(self) -> torch.Tensor:
        meta, data = self.recv_fwd_sock.recv_multipart()
        return self._deserialize(meta, data)

    def send_backward(self, grad_tensor: torch.Tensor):
        self.send_bwd_sock.send_multipart(self._serialize(grad_tensor))

    def recv_backward(self) -> torch.Tensor:
        meta, data = self.recv_bwd_sock.recv_multipart()
        return self._deserialize(meta, data)

    def close(self):
        self.context.term()
