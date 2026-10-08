"""Pipelined forward-only inference across engine stages.

Latency mode (`ack_each=True`): the terminal stage acks every batch back along the
backward sockets, so stage 0 measures true end-to-end round-trip time on ONE clock.
Throughput mode (`ack_each=False`): batches stream through the pipeline, and a single
ack at the end lets stage 0 time the whole pass (including pipeline fill/drain).
"""
import time

import torch
import torch.nn.functional as F


def run_inference(stage_id, num_stages, model, transceiver, num_batches, batches=None, labels=None,
                  ack_each=False):
    """All stages must call this with the same num_batches/ack_each.

    batches: iterator of input tensors (stage 0 only).
    labels:  list of label tensors, one per batch (terminal stage only).
    """
    last = num_stages - 1
    ack = torch.zeros(1)
    model.eval()
    result = {}
    with torch.no_grad():
        if stage_id == 0:
            latencies = []
            t_start = time.perf_counter()
            for _ in range(num_batches):
                x = next(batches)
                t0 = time.perf_counter()
                transceiver.send_forward(model(x))
                if ack_each:
                    transceiver.recv_backward()
                    latencies.append(time.perf_counter() - t0)
            if not ack_each:
                transceiver.recv_backward()
            result = {"elapsed_s": time.perf_counter() - t_start, "latencies_s": latencies}
        elif stage_id < last:
            for _ in range(num_batches):
                transceiver.send_forward(model(transceiver.recv_forward()))
                if ack_each:
                    transceiver.send_backward(transceiver.recv_backward())
            if not ack_each:
                transceiver.send_backward(transceiver.recv_backward())
        else:
            correct, total, loss_sum = 0, 0, 0.0
            for i in range(num_batches):
                out = model(transceiver.recv_forward())
                y = labels[i]
                correct += (out.argmax(1) == y).sum().item()
                loss_sum += F.cross_entropy(out, y, reduction="sum").item()
                total += len(y)
                if ack_each:
                    transceiver.send_backward(ack)
            if not ack_each:
                transceiver.send_backward(ack)
            result = {"correct": correct, "total": total, "loss_sum": loss_sum}
    return result
