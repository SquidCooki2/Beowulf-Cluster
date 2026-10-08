"""Merge benchmark runs (engine + HPC CPU/GPU) into a comparison table and plot.

    python -m benchmark.compare --results-dir results --target-acc 0.9

Each run lives in <results-dir>/<run-id>/. HPC runs write train.json / infer.json; engine
runs write train_stage{i}.json / infer_stage{i}.json (merged here: timing from stage 0's
clock, accuracy from the terminal stage).
"""
import argparse
import glob
import json
import os
import statistics


def _read(path):
    with open(path) as f:
        return json.load(f)


def _merge_engine(docs):
    s0 = next(d for d in docs if d["stage"] == 0)
    sl = next(d for d in docs if d["stage"] == d["num_stages"] - 1)
    out = {**s0, "stages": sorted(docs, key=lambda d: d["stage"]), "system": s0["system"]}
    out["peak_rss_mb"] = max(d.get("peak_rss_mb", 0) for d in docs)
    if s0["kind"] == "train":
        out["epochs"] = [
            {**e0, "train_loss": el["train_loss"], "val_acc": el["val_acc"], "val_loss": el["val_loss"]}
            for e0, el in zip(s0["epochs"], sl["epochs"])]
        out["test"] = sl.get("test")
    else:
        lat = dict(s0.get("latency_b1", {}))
        lat["acc"] = sl.get("latency_b1", {}).get("acc")
        out["latency_b1"] = lat
        out["throughput"] = {b: {**v, "acc": sl["throughput"][b]["acc"]} for b, v in s0["throughput"].items()}
    return out


def load_runs(results_dir):
    """-> list of {"run_id", "platform", "train": doc|None, "infer": doc|None}"""
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(run_dir):
            continue
        entry = {"run_id": os.path.basename(run_dir), "train": None, "infer": None}
        for kind in ("train", "infer"):
            single = os.path.join(run_dir, f"{kind}.json")
            stage_files = glob.glob(os.path.join(run_dir, f"{kind}_stage*.json"))
            if os.path.exists(single):
                entry[kind] = _read(single)
            elif stage_files:
                entry[kind] = _merge_engine([_read(f) for f in stage_files])
        doc = entry["train"] or entry["infer"]
        if doc:
            entry["platform"] = doc["platform"]
            runs.append(entry)
    return runs


def describe(doc):
    s = doc["system"]
    if "stages" in doc:
        return f"{len(doc['stages'])} nodes ({', '.join(sorted({d['system']['cpu'] for d in doc['stages']}))})"
    return s.get("gpu") or f"{s['cpu']} ({s['torch_threads']} threads)"


def time_to_target(epochs, target):
    elapsed = 0.0
    for e in epochs:
        elapsed += e["train_s"] + e["val_s"]
        if e["val_acc"] >= target:
            return elapsed
    return None


def summarize(run, target):
    row = {"run_id": run["run_id"], "platform": run["platform"]}
    t, i = run["train"], run["infer"]
    if t:
        ep = t["epochs"]
        row.update(hardware=describe(t),
                   s_per_epoch=statistics.median(e["train_s"] + e["val_s"] for e in ep),
                   train_img_s=statistics.median(e["images_per_s"] for e in ep),
                   t_target=time_to_target(ep, target), best_val_acc=max(e["val_acc"] for e in ep),
                   test_acc=(t.get("test") or {}).get("acc"), peak_rss_mb=t.get("peak_rss_mb"),
                   epochs=len(ep))
    if i:
        row.setdefault("hardware", describe(i))
        row.update(lat_p50=i["latency_b1"]["p50_ms"], lat_p95=i["latency_b1"]["p95_ms"],
                   infer_img_s={b: v["images_per_s"] for b, v in i["throughput"].items()})
    return row


def fmt(v, spec=".2f", none="n/a"):
    return none if v is None else format(v, spec)


def fmtx(v):
    return "n/a" if v is None else f"{v:.1f}x"


def ratio(a, b):
    return None if a is None or b is None or b == 0 else a / b


def build_report(rows, baseline, target):
    L = [f"# Benchmark comparison (target val acc = {target})", ""]
    L += ["| Platform | Hardware | s/epoch | train img/s | time to target (s) | best val acc | test acc | "
          "lat p50 (ms) | lat p95 (ms) | infer img/s (b=32) |", "|" + "---|" * 10]
    for r in rows:
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["platform"], r.get("hardware", ""), fmt(r.get("s_per_epoch"), ".1f"), fmt(r.get("train_img_s"), ".1f"),
            fmt(r.get("t_target"), ".0f", "not reached"), fmt(r.get("best_val_acc"), ".4f"),
            fmt(r.get("test_acc"), ".4f"), fmt(r.get("lat_p50")), fmt(r.get("lat_p95")),
            fmt((r.get("infer_img_s") or {}).get("32"), ".1f")))
    base = next((r for r in reversed(rows) if r["platform"] == baseline), None)
    if base:
        L += ["", f"## Speedup vs `{baseline}` (>1 means faster than the engine)", "",
              "| Platform | training throughput | time to target | latency p50 | inference b=1 | b=8 | b=32 |",
              "|" + "---|" * 7]
        for r in rows:
            if r is base:
                continue
            ib, ir = base.get("infer_img_s") or {}, r.get("infer_img_s") or {}
            L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                r["platform"], fmtx(ratio(r.get("train_img_s"), base.get("train_img_s"))),
                fmtx(ratio(base.get("t_target"), r.get("t_target"))),
                fmtx(ratio(base.get("lat_p50"), r.get("lat_p50"))),
                *[fmtx(ratio(ir.get(b), ib.get(b))) for b in ("1", "8", "32")]))
    else:
        L += ["", f"_No `{baseline}` run found, so no speedups were computed._"]
    return "\n".join(L) + "\n"


def plot(runs, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in runs:
        if not r["train"]:
            continue
        t, xs, ys = 0.0, [], []
        for e in r["train"]["epochs"]:
            t += e["train_s"] + e["val_s"]
            xs.append(t)
            ys.append(e["val_acc"])
        ax.plot(xs, ys, marker="o", label=f"{r['platform']} ({r['run_id']})")
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("validation accuracy")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--target-acc", type=float, default=0.9)
    p.add_argument("--baseline", default="engine", help="Platform label speedups are computed against")
    args = p.parse_args()
    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No runs found in {args.results_dir}")
    report = build_report([summarize(r, args.target_acc) for r in runs], args.baseline, args.target_acc)
    print(report)
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write(report)
    plot(runs, os.path.join(args.results_dir, "accuracy_vs_time.png"))


if __name__ == "__main__":
    main()
