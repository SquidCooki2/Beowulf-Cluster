"""Inference benchmark for the HPC / single-device side (counterpart of `run_stage.py --mode infer`).

Measures batch-1 latency (p50/p95) and whole-test-set throughput at several batch sizes,
timing host->device copy + forward + result read-back, like a real request would.

    python -m benchmark.infer_local --device cuda --weights-dir $SCRATCH_DIR/output/malaria_mobilenet
"""
import argparse
import os
import time

import torch
import torch.nn as nn

from benchmark.metrics import latency_summary, peak_rss_mb, save_json, sync, system_info
from data import malaria
from models.registry import DEFAULT_MODEL, get_spec

DEFAULT_SCRATCH = os.environ.get("SCRATCH_DIR", f"/scratch/{os.environ.get('USER', 'username')}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data-dir", default=os.path.join(DEFAULT_SCRATCH, "data"))
    p.add_argument("--weights-dir", default=None, help="Dir with model.pt from training.train_local")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--platform-label", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--num-stages", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--latency-samples", type=int, default=200)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    if args.threads:
        torch.set_num_threads(args.threads)
    spec = get_spec(args.model)
    model = nn.Sequential(*spec.create_stages(args.num_stages, pretrained=not args.no_pretrained))
    if args.weights_dir:
        model.load_state_dict(torch.load(os.path.join(args.weights_dir, "model.pt"), map_location="cpu"))
    model.to(device).eval()

    x, y = malaria.load_arrays(args.data_dir)
    _, _, te = malaria.split_indices(len(y), limit=args.limit)

    def run(chunks):
        """Time each chunk (copy + forward + argmax readback); returns per-chunk seconds and #correct."""
        times, correct = [], 0
        with torch.no_grad():
            for c in chunks:
                batch = malaria.to_tensor(x[c])           # preprocessing excluded from timing
                labels = torch.from_numpy(y[c])
                sync(device)
                t0 = time.perf_counter()
                pred = model(batch.to(device)).argmax(1).cpu()
                sync(device)
                times.append(time.perf_counter() - t0)
                correct += (pred == labels).sum().item()
        return times, correct

    # Latency (batch 1)
    lat_ids = te[:args.latency_samples]
    run([lat_ids[i:i + 1] for i in range(min(10, len(lat_ids)))])  # warmup
    times, correct = run([lat_ids[i:i + 1] for i in range(len(lat_ids))])
    record = {"kind": "infer", "platform": args.platform_label or f"hpc-{device.type}", "run_id": args.run_id,
              "model": spec.name, "system": system_info(device),
              "config": {"limit": args.limit, "weights": args.weights_dir, "pretrained": not args.no_pretrained},
              "latency_b1": {**latency_summary(times), "acc": correct / len(lat_ids)}, "throughput": {}}
    print(f"[latency b=1] p50={record['latency_b1']['p50_ms']:.2f}ms p95={record['latency_b1']['p95_ms']:.2f}ms")

    # Throughput over the whole test split
    for b in args.batch_sizes:
        chunks = [te[i:i + b] for i in range(0, len(te), b)]
        run(chunks[:2])  # warmup
        t0 = time.perf_counter()
        times, correct = run(chunks)
        elapsed = time.perf_counter() - t0  # end-to-end incl. preprocessing, like the engine's stage 0
        record["throughput"][str(b)] = {"images_per_s": len(te) / elapsed, "elapsed_s": elapsed,
                                        "n": len(te), "acc": correct / len(te)}
        print(f"[throughput b={b}] {len(te) / elapsed:.1f} img/s acc={correct / len(te):.4f}")
    record["peak_rss_mb"] = peak_rss_mb()
    save_json(os.path.join(args.results_dir, args.run_id, "infer.json"), record)


if __name__ == "__main__":
    main()
