"""Single-device trainer for the registered model (HPC GPU/CPU node or laptop).

The model is the engine's stage list joined into one nn.Sequential, so weights
trained here load directly into the pipeline engine (per-stage files are saved).
Per-epoch timings and accuracy are written to JSON for comparison with the engine.

    python -m training.train_local --model malaria_mobilenet --device cuda --recipe full
"""
import argparse
import importlib
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmark.metrics import peak_rss_mb, save_json, sync, system_info
from models.registry import DEFAULT_MODEL, REGISTRY, get_spec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(REGISTRY))
    p.add_argument("--data-dir", default=os.path.join(os.getcwd(), "data"),
                   help="Directory for datasets (default: ./data)")
    p.add_argument("--output-dir", default=os.path.join(os.getcwd(), "output"),
                   help="Directory for saved model weights (default: ./output)")
    p.add_argument("--results-dir", default="results", help="Benchmark JSON goes to <results-dir>/<run-id>/")
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--platform-label", default=None, help="Default: hpc-<device>")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    p.add_argument("--num-stages", type=int, default=5, help="Stage split (must match the engine for weight reuse)")
    p.add_argument("--recipe", default="full", choices=["full", "two_phase"],
                   help="full: fine-tune every layer (used for engine comparison); "
                        "two_phase: last stage first, then everything at a low LR")
    p.add_argument("--epochs", type=int, default=10, help="Epochs (phase 1 for two_phase)")
    p.add_argument("--finetune-epochs", type=int, default=10, help="Phase 2 epochs (two_phase only)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--finetune-lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Use N samples total (smoke tests)")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--allow-home", action="store_true", help="Disable the home-directory safety check")
    return p.parse_args(argv)


def check_dirs(args):
    home = os.path.expanduser("~")
    for d in (args.data_dir, args.output_dir):
        ap = os.path.abspath(d)
        if not args.allow_home and (ap.startswith("/users") or ap.startswith(home)):
            print(f"[NOTE] Path '{ap}' is located in a home/user directory. "
                  "Packed .npz format is safe, but ensure loose files don't exceed your inode quota. "
                  "Use scratch if you run large multi-file experiments.", flush=True)
        os.makedirs(ap, exist_ok=True)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    sync(device)
    t0 = time.perf_counter()
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)  # .item() also syncs the GPU
            n += len(y)
            correct += (out.argmax(1) == y).sum().item()
    sync(device)
    return {"loss": total_loss / n, "acc": correct / n, "seconds": time.perf_counter() - t0, "n": n}


def fit(model, loaders, device, epochs, lr, params, tag, loss_fn, log, start_epoch=0):
    optimizer = torch.optim.Adam(params, lr=lr)
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, loaders[0], loss_fn, device, optimizer)
        va = run_epoch(model, loaders[1], loss_fn, device)
        rec = {"epoch": start_epoch + ep, "phase": tag, "train_s": tr["seconds"], "val_s": va["seconds"],
               "images_per_s": tr["n"] / tr["seconds"], "train_loss": tr["loss"], "train_acc": tr["acc"],
               "val_loss": va["loss"], "val_acc": va["acc"]}
        log.append(rec)
        print(f"[{tag}] epoch {rec['epoch']} train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
              f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f} "
              f"train_s={tr['seconds']:.1f} ({rec['images_per_s']:.1f} img/s)", flush=True)
    return start_epoch + epochs


def main(argv=None):
    args = parse_args(argv)
    check_dirs(args)
    spec = get_spec(args.model)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    print(f"Model={spec.name} device={device} recipe={args.recipe}")

    data_mod = importlib.import_module(f"data.{spec.data}")
    train_ds, val_ds, test_ds = data_mod.load(args.data_dir, limit=args.limit)
    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
                                   drop_last=sh)  # drop_last on train: matches the engine's fixed-size batches
    loaders = (mk(train_ds, True), mk(val_ds, False), mk(test_ds, False))

    stages = spec.create_stages(args.num_stages, pretrained=not args.no_pretrained)
    model = nn.Sequential(*stages).to(device)
    loss_fn = nn.CrossEntropyLoss()
    log = []

    if args.recipe == "full":
        fit(model, loaders, device, args.epochs, args.lr, model.parameters(), "full", loss_fn, log)
    else:
        for p in model.parameters():
            p.requires_grad = False
        head = list(stages[-1].parameters())
        for p in head:
            p.requires_grad = True
        done = fit(model, loaders, device, args.epochs, args.lr, head, "head", loss_fn, log)
        for p in model.parameters():  # FrozenBatchNorm has no params, so it stays frozen
            p.requires_grad = True
        fit(model, loaders, device, args.finetune_epochs, args.finetune_lr,
            [p for p in model.parameters() if p.requires_grad], "finetune", loss_fn, log, done)

    test = run_epoch(model, loaders[2], loss_fn, device)
    print(f"[test] loss={test['loss']:.4f} acc={test['acc']:.4f}")

    out = os.path.join(args.output_dir, spec.name)
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    for i, s in enumerate(stages):  # per-stage files for the pipeline engine
        torch.save(s.state_dict(), os.path.join(out, f"stage{i}.pt"))
    print(f"Saved weights to {out}")

    save_json(os.path.join(args.results_dir, args.run_id, "train.json"), {
        "kind": "train", "platform": args.platform_label or f"hpc-{device.type}", "run_id": args.run_id,
        "model": spec.name, "system": system_info(device), "peak_rss_mb": peak_rss_mb(),
        "config": {"recipe": args.recipe, "batch_size": args.batch_size, "lr": args.lr,
                   "epochs": args.epochs, "num_stages": args.num_stages, "limit": args.limit,
                   "pretrained": not args.no_pretrained, "seed": args.seed},
        "epochs": log, "test": {"loss": test["loss"], "acc": test["acc"], "n": test["n"]},
    })


if __name__ == "__main__":
    main()
