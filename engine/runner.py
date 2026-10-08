"""Real-data training / inference drivers for one pipeline stage (used by run_stage.py).

Every stage runs the same loops with identical counts; stage 0 feeds images and the
terminal stage reads labels from the shared, deterministic dataset split, so no data
is exchanged except activations/gradients. Stage 0's clock gives end-to-end timing.
"""
import logging
import os
import time

import numpy as np
import torch

from benchmark.metrics import latency_summary, peak_rss_mb, save_json, system_info
from data import malaria
from engine.inference import run_inference


def _load(stage_id, args):
    # Only stage 0 needs pixels (~1.3GB); everyone else just reads the tiny label array.
    x, y = malaria.load_arrays(args.data_dir, images=(stage_id == 0))
    tr, va, te = malaria.split_indices(len(y), seed=0, limit=args.limit)
    return x, y, tr, va, te


def _chunks(idx, size):
    return [idx[i:i + size] for i in range(0, len(idx), size)]


def _pass(stage_id, num_stages, model, transceiver, x, y, chunks, ack_each):
    """One forward-only pass over `chunks` (lists of dataset indices)."""
    batches = (malaria.to_tensor(x[c]) for c in chunks) if stage_id == 0 else None
    labels = [torch.from_numpy(y[c]) for c in chunks] if stage_id == num_stages - 1 else None
    return run_inference(stage_id, num_stages, model, transceiver, len(chunks), batches, labels, ack_each)


def _base_record(stage_id, num_stages, kind, args, config):
    return {
        "kind": kind, "platform": args.platform_label, "run_id": args.run_id,
        "stage": stage_id, "num_stages": num_stages, "model": args.model_name,
        "system": system_info("cpu"),
        "config": {"micro_batch_size": config["micro_batch_size"],
                   "num_microbatches": config["num_microbatches"],
                   "batch_size": config["micro_batch_size"] * config["num_microbatches"],
                   "lr": config["learning_rate"], "limit": args.limit, "seed": args.seed,
                   "pretrained": not args.no_pretrained, "epochs": args.epochs,
                   "devices": [s["device_type"] for s in config["stages"]]},
    }


def run_train(stage_id, num_stages, model, scheduler, transceiver, config, args):
    last = num_stages - 1
    mb, nmb = config["micro_batch_size"], config["num_microbatches"]
    bs = mb * nmb
    x, y, tr, va, te = _load(stage_id, args)
    iters = len(tr) // bs
    if iters == 0:
        raise ValueError(f"Train split ({len(tr)}) smaller than one batch ({bs}); raise --limit")
    aug_rng = np.random.default_rng(args.seed + 100)
    log = []

    for epoch in range(args.epochs):
        order = malaria.epoch_order(tr, epoch, args.seed)
        before = transceiver.stats()
        scheduler.loss_history.clear()
        model.train()
        t0 = time.perf_counter()
        for it in range(iters):
            ids = order[it * bs:(it + 1) * bs]
            micro = target = None
            if stage_id == 0:
                micro = iter([malaria.to_tensor(x[ids[j * mb:(j + 1) * mb]], augment=True, rng=aug_rng)
                              for j in range(nmb)])
            if stage_id == last:
                target = iter([torch.from_numpy(y[ids[j * mb:(j + 1) * mb]]) for j in range(nmb)])
            scheduler.run_1f1b_epoch(dataloader_iterator=micro, target_iterator=target)
        train_s = time.perf_counter() - t0
        after = transceiver.stats()

        t1 = time.perf_counter()
        val = _pass(stage_id, num_stages, model, transceiver, x, y, _chunks(va, bs), ack_each=False)
        val_s = time.perf_counter() - t1

        rec = {"epoch": epoch + 1, "train_s": train_s, "val_s": val_s,
               "images_per_s": iters * bs / train_s,
               "comm": {k: after[k] - before[k] for k in after}}
        if stage_id == last:
            rec.update(train_loss=float(np.mean(scheduler.loss_history)),
                       val_acc=val["correct"] / val["total"], val_loss=val["loss_sum"] / val["total"])
        log.append(rec)
        logging.info(f"Stage {stage_id} epoch {epoch + 1}/{args.epochs}: train_s={train_s:.1f} "
                     f"({rec['images_per_s']:.1f} img/s) recv_wait={rec['comm']['recv_wait_s']:.1f}s"
                     + (f" val_acc={rec['val_acc']:.4f} train_loss={rec['train_loss']:.4f}"
                        if stage_id == last else ""))

    test = _pass(stage_id, num_stages, model, transceiver, x, y, _chunks(te, bs), ack_each=False)
    record = _base_record(stage_id, num_stages, "train", args, config)
    record.update(epochs=log, peak_rss_mb=peak_rss_mb())
    if stage_id == last:
        record["test"] = {"acc": test["correct"] / test["total"], "loss": test["loss_sum"] / test["total"],
                          "n": test["total"]}
        logging.info(f"Test acc={record['test']['acc']:.4f}")
    run_dir = os.path.join(args.results_dir, args.run_id)
    save_json(os.path.join(run_dir, f"train_stage{stage_id}.json"), record)
    wdir = os.path.join(run_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(wdir, f"stage{stage_id}.pt"))


def run_infer(stage_id, num_stages, model, transceiver, config, args):
    last = num_stages - 1
    x, y, _, _, te = _load(stage_id, args)
    record = _base_record(stage_id, num_stages, "infer", args, config)

    # Latency: batch size 1, acked per sample (round-trip on stage 0's clock)
    lat_ids = te[:args.latency_samples]
    _pass(stage_id, num_stages, model, transceiver, x, y, _chunks(lat_ids[:10], 1), ack_each=True)  # warmup
    res = _pass(stage_id, num_stages, model, transceiver, x, y, _chunks(lat_ids, 1), ack_each=True)
    if stage_id == 0:
        record["latency_b1"] = latency_summary(res["latencies_s"])
    if stage_id == last:
        record["latency_b1"] = {"acc": res["correct"] / res["total"]}

    # Throughput: pipelined batches over the whole test split
    record["throughput"] = {}
    for b in args.infer_batch_sizes:
        _pass(stage_id, num_stages, model, transceiver, x, y, _chunks(te[:2 * b], b), ack_each=False)  # warmup
        res = _pass(stage_id, num_stages, model, transceiver, x, y, _chunks(te, b), ack_each=False)
        if stage_id == 0:
            record["throughput"][str(b)] = {"images_per_s": len(te) / res["elapsed_s"],
                                            "elapsed_s": res["elapsed_s"], "n": len(te)}
        if stage_id == last:
            record["throughput"][str(b)] = {"acc": res["correct"] / res["total"]}
        logging.info(f"Stage {stage_id}: inference batch={b} done")
    record["comm"] = transceiver.stats()
    record["peak_rss_mb"] = peak_rss_mb()
    save_json(os.path.join(args.results_dir, args.run_id, f"infer_stage{stage_id}.json"), record)
