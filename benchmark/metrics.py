"""Shared measurement helpers so the engine and HPC runs are timed and recorded identically."""
import json
import os
import platform
import socket
import subprocess
import time

import torch

time_now = time.perf_counter


def _cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def system_info(device):
    info = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
        "commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if torch.device(device).type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def percentile(values, p):
    """Linear-interpolated percentile (p in [0, 100])."""
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def latency_summary(latencies_s):
    ms = [x * 1000.0 for x in latencies_s]
    return {"n": len(ms), "mean_ms": sum(ms) / len(ms), "p50_ms": percentile(ms, 50),
            "p95_ms": percentile(ms, 95), "p99_ms": percentile(ms, 99)}


def peak_rss_mb():
    try:
        import resource  # Linux / macOS
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024.0 if platform.system() == "Linux" else peak / (1024.0 ** 2)
    except ImportError:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 ** 2)


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Wrote {path}", flush=True)
