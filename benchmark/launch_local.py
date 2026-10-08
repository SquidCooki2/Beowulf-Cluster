"""Run all pipeline stages as local processes on 127.0.0.1 (dev / smoke tests, no cluster needed).

    python -m benchmark.launch_local -- --mode synthetic
    python -m benchmark.launch_local --num-stages 3 -- --mode train --data-dir ./tmp --limit 256 --epochs 1

Everything after `--` is forwarded to run_stage.py on every stage.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.json")
    p.add_argument("--num-stages", type=int, default=None, help="Use only the first N stages of the config")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("stage_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    extra = [a for a in args.stage_args if a != "--"]

    with open(args.config) as f:
        config = json.load(f)
    config = copy.deepcopy(config)
    if args.num_stages:
        config["stages"] = config["stages"][:args.num_stages]
    for s in config["stages"]:
        s["ip"] = "127.0.0.1"
    n = len(config["stages"])

    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(config, tmp)
    tmp.close()

    env = dict(os.environ, PYTHONPATH=os.getcwd())
    procs = {}
    try:
        for sid in range(n - 1, -1, -1):  # workers first so sockets are bound
            procs[sid] = subprocess.Popen(
                [sys.executable, "run_stage.py", "--stage", str(sid), "--config", tmp.name, *extra], env=env)
            if sid == 1:
                time.sleep(2)
        deadline = time.time() + args.timeout
        for sid, pr in procs.items():
            pr.wait(timeout=max(1, deadline - time.time()))
    except subprocess.TimeoutExpired:
        print("Timed out; killing stages")
    finally:
        for pr in procs.values():
            if pr.poll() is None:
                pr.kill()
        os.unlink(tmp.name)
    codes = {sid: pr.returncode for sid, pr in sorted(procs.items())}
    print("exit codes:", codes)
    sys.exit(0 if all(c == 0 for c in codes.values()) else 1)


if __name__ == "__main__":
    main()
