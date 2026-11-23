# Benchmark comparison (target val acc = 0.95)

| Platform | Hardware | s/epoch | train img/s | time to target (s) | best val acc | test acc | lat p50 (ms) | lat p95 (ms) | infer img/s (b=32) |
|---|---|---|---|---|---|---|---|---|---|
| engine | 3 nodes (Intel(R) Core(TM) i5-9500T CPU @ 2.20GHz) | 208.0 | 113.2 | 211 | 0.9724 | 0.9608 | 9.82 | 10.96 | 205.7 |

## Speedup vs `engine` (>1 means faster than the engine)

| Platform | training throughput | time to target | latency p50 | inference b=1 | b=8 | b=32 |
|---|---|---|---|---|---|---|
