# Measuring What Matters: Pitfalls in Benchmarking Data and Model Parallelism on Commodity GPUs

Code, raw measurements and analysis for the paper of the same name.

We benchmark data parallelism (DP) and naive two-stage model parallelism (MP) for five vision models on two PCIe-connected NVIDIA T4 GPUs, and measure how much eight common measurement choices change the result. Every flawed and correct value in the paper is computed from the same recorded training steps.

**Paper:** _add link (arXiv / proceedings) here_

## Main results

| | Flawed | Correct |
|---|---|---|
| DP speedup with the DDP batch-size mismatch (P1) | 0.98× | 1.90× |
| DP communication with the optimizer step included (P2) | 9.02 ms | 6.33 ms |
| MP stage imbalance from forward-only timing (P5) | 0.71–0.82 for every model | 0.01–0.54 |
| Naive MP speedup over one GPU (P7) | 1.07–1.30× | 0.97–1.01× (power-neutral) |
| Step time on a cold GPU (P8) | 89.9 ms | 100.7 ms |

Measured correctly, DP reaches 1.93–2.01× at a global batch of 64 for all five models.

## Repository layout

```
notebooks/
  parallelism_benchmark.ipynb   main benchmark: 1-GPU, DP, MP, all-reduce test (one repeat per run)
  parallelism_extra.ipynb       cold start, duty cycle, 128x128 runs, NCCL transport test
  parallelism_analysis.ipynb    builds every table and figure in the paper (CPU only)
data/
  repeat1/  repeat2/  repeat3/  output of the main notebook, one folder per repeat
  extra/                        output of the extra-experiments notebook
results/                        tables (CSV, LaTeX) and figures (PDF, PNG) produced by the analysis
paper/                          LaTeX source of the paper
```

## Hardware and software

All runs used Kaggle notebooks with **GPU T4 x2**: two NVIDIA T4 GPUs (16 GB, 70 W power limit) connected over PCIe through the CPU host bridge, 4 CPU cores.

| Package | Version |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0 (CUDA 12.8) |
| NCCL | 2.31.2 (runtime) |

Each run saves its own environment record (`environment-r*.json` or `environment.json`) with library versions, GPU names, the CUDA peer-access flag and the `nvidia-smi topo -m` output.

The notebooks use synthetic inputs, so no dataset download is needed.

## How to reproduce

### 1. Main benchmark (about 1 h 45 min per repeat)

1. Import `notebooks/parallelism_benchmark.ipynb` into a new Kaggle notebook.
2. Set **Accelerator → GPU T4 x2**.
3. In the first cell, set `REPEAT = 1`.
4. Click **Save Version → Save & Run All**. The run continues if you close the browser.
5. When it finishes, change to `REPEAT = 2`, save and run again; then `REPEAT = 3`.

Each repeat runs in a new session, which on Kaggle usually means a different machine. That is intended: the spread between repeats includes machine-to-machine variation.

After the first successful run, set the notebook's environment to **Pin to original environment** so that all repeats use the same software.

### 2. Extra experiments (about 1 h 30 min)

Import `notebooks/parallelism_extra.ipynb` into a **separate** notebook, set **GPU T4 x2**, and run it with **Save & Run All**. It must start on a fresh, cool GPU, because the first experiment measures a cold start.

### 3. Analysis (a few minutes, no GPU)

1. Import `notebooks/parallelism_analysis.ipynb` and set **Accelerator → None**.
2. Attach the data: either the `data/` folder of this repository uploaded as a Kaggle dataset, or the outputs of your own runs.
3. Click **Run All**. The first cell should report `repeats found: [1, 2, 3]`, 50 result files per repeat and 34 extra-experiment files.

All tables and figures are written to `/kaggle/working/paper/`. The notebook also runs locally with `pandas`, `numpy` and `matplotlib` (plus `torch` and `torchvision` for the FLOP counts) if you point it at the `data/` folder.

## Where each table and figure comes from

| Paper | Analysis output |
|---|---|
| Table: machines used | `tab_environment` |
| Table: models | `tab_models`, `tab_baseline` |
| Table: data parallelism | `tab_dp` |
| Table: naive two-stage MP | `tab_mp` |
| Table: duty cycle | `tab_duty_cycle` |
| Table: size of each pitfall | `tab_pitfalls` (all cases in `pitfalls_all_cases.csv`) |
| Table: lower resolution | `tab_resolution` |
| Fig. 1: measurement design | `paper/image/fig_overview.pdf` (TikZ source in `paper/`) |
| Fig.: speedup | `fig_speedup` |
| Fig.: all-reduce bandwidth | `fig_allreduce_bandwidth` |
| Fig.: duty cycle | `fig_duty_cycle` |
| Fig.: cold start | `fig_coldstart_resnet18` |

## Data format

One CSV file per run, one row per measured training step (100 steps per run).

**File names**

```
{model}-{config}-bs{global batch}-res{resolution}-r{repeat}.csv   main runs
{model}-{config}-bs{batch}-res{resolution}.csv                    extra experiments
{same name}-clocks.csv                                            GPU log for that run
allreduce-r{repeat}.csv, allreduce-default.csv, allreduce-p2p_forced.csv
```

`model` is one of `resnet18`, `resnet34`, `resnet50`, `vittiny`, `vitbase`. `config` is `1gpu`, `dp`, `mp`, `coldstart` or `dutyN` (idle gap of N% of the step time).

**Columns** (all times in milliseconds)

| Config | Main columns |
|---|---|
| `1gpu` | `fwdbwd_ms`, `opt_ms`, `total_ms`, `peak_mem_mb` |
| `dp` | `per_gpu_batch`, `order`, `nosync_ms`, `sync_ms`, `opt_ms`, `step_ms`, `exposed_comm_ms`, `nccl_ms`, `peak_mem_mb` |
| `mp` | `fwd0_ms`, `xfer_fwd_ms`, `fwd1_ms`, `bwd1_ms`, `xfer_bwd_ms`, `bwd0_ms`, `opt_ms`, `staged_total_ms`, `step_ms`, `step_sync0_ms`, `act_mb`, `peak0_mb`, `peak1_mb` |
| `coldstart` | `step`, `wall`, `t_s`, `step_ms` |
| `dutyN` | `gap_ratio`, `busy_ms`, `gap_ms` |
| `allreduce-*` | `size_mb`, `median_ms`, `gb_per_s` |

Notes:

- In `dp` files, `batch` is the **global** batch; each GPU processes `per_gpu_batch` = batch / 2. `exposed_comm_ms` = `sync_ms` − `nosync_ms` and can be negative; no rows were removed. `nccl_ms` is the total NCCL kernel time per step from 20 separately profiled steps.
- In `mp` files, `step_ms` is the normal step timed after synchronizing both GPUs, and `step_sync0_ms` the same step timed after synchronizing GPU 0 only. The seven phase columns come from the same step split into timed phases.
- Clock files have no header. Columns: timestamp, GPU index, SM clock (MHz), power (W), temperature (°C), sampled every 200 ms.

## Known limitations

- One GPU type (T4), two GPUs, PCIe only. The power and thermal effects depend on the T4's 70 W limit.
- MP is a two-stage sequential split without micro-batches, not an optimized pipeline.
- Three repeats, so confidence intervals are wide.
- Clock samples every 200 ms are too coarse to resolve individual MP phases.

See the Threats to Validity section of the paper for details.


## License

_Add a license here, for example MIT for the code and CC BY 4.0 for the data._
