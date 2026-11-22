# IEEE Paper (`paper/main.tex`) — Gap & Figure Checklist

This directory contains the IEEE two-column LaTeX source (`IEEEtran`) for the project:

- `main.tex` — Complete 6–8 page IEEE conference draft (Overleaf-ready).
- `references.bib` — BibTeX citations.

All missing empirical numbers, hardware specs, and charts are highlighted in red using `\gap{ID: ...}` and `\placeholderfig{...}` / `\placeholderfigwide{...}`.

---

## 1. Quick Gap Reference

| Gap ID(s) | Where to Find / How to Measure |
| :--- | :--- |
| `ABS-1` .. `ABS-4`, `CONC-1` | Headline summary numbers filled in after Tables III–V are complete |
| `HW-1` .. `HW-20` | `lscpu`, `free -h`, `nvidia-smi` (HPC), kill-a-watt / `powertop` wall-power readings, estimated acquisition/refurb cost |
| `PART-1` .. `PART-9` | Startup log from `run_stage.py`: `FLOPS-weighted partition: gflops={...} -> layers=[...]` |
| `TR-1` .. `TR-48` | `python -m benchmark.compare --results-dir results --target-acc 0.9` (reads `train.json` / `train_stage*.json`) |
| `INF-1` .. `INF-40` | `python -m benchmark.compare --results-dir results` (reads `infer.json` / `infer_stage*.json`) |
| `COM-1` .. `COM-9` | `TensorTransceiver.stats()` fields (`recv_wait_s`, `send_s`, `bytes_sent`, `bytes_recv`) in `train_stage*.json` |
| `ENG-1` .. `ENG-4` | Wall-power meter (Watts) $\times$ epoch duration (`s/epoch / 3600`) |
| `EW-1` .. `EW-3` | Chassis weights of repurposed HP ProDesk 400 G5s + laptop (kg of e-waste diverted) |

---

## 2. Planned Figures (`paper/figures/`)

1. **`fig1_topology`** — Physical `10.0.0.0/24` topology (3 ProDesks + 4th generic laptop + Ansible + NFS `/shared`) & 1F1B 8-micro-batch timeline.
2. **`fig2_flops_partition`** — Node GFLOPS vs. allocated layers (static vs. FLOPS-weighted).
3. **`fig3_convergence`** — Validation accuracy & training loss vs. wall-clock time (`malaria_mobilenet` and `scienceqa_smolvlm`).
4. **`fig4_speedup`** — Throughput speedup chart from `benchmark/compare.py`.
5. **`fig5_breakdown`** — Per-stage time breakdown (Compute vs. `send_s` vs. `recv_wait_s`) on the 4-node heterogeneous cluster.
6. **`fig6_memory_energy`** — Per-stage `peak_rss_mb` ($O(p)$ memory bound) and Wh/epoch energy comparison.
