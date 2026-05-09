# vLLM CPU Performance Optimization Skill

## When to Use This Skill

Read and follow this skill when the user asks to:
- "make vLLM faster", "optimize tokens per second", "run the perf loop", "benchmark inference"
- improve throughput on Apple Silicon / macOS

**Read this entire file before executing any step.**

---

## Platform Constraints

- vLLM has **no MPS backend** on macOS. The only valid device is `--device cpu`.
- All optimizations in this skill are CPU-compatible only.
- Benchmarking (`--output-json`) and profiling (`--profile`) are **mutually exclusive** in one invocation — always run them as separate commands.

---

## Constants (defaults — do not change without user instruction)

```
MODEL              = Qwen/Qwen2.5-0.5B-Instruct
BATCH_SIZE         = 8        # Apple Silicon CPU-appropriate (not 8×128 which is too slow)
INPUT_LEN          = 32       # reduced from 128 — keeps each iter under 2 min on CPU
OUTPUT_LEN         = 32       # reduced from 128 — same reason
GPU_MEM_UTIL       = 0.3      # CRITICAL: default 0.92 OOMs on 16GB Mac with ~5GB free
MAX_ITERS          = 5
RESULTS_DIR        = perf_results   # workspace-relative — /tmp is not shared across shells
BRANCH             = perf/vllm-cpu-opt
LOG_FILE           = docs/perf/cpu_opt_log.md
SKILL_DIR          = .cursor/skills/vllm-perf
```

**Correct tokens/sec formula (critical):**
```
tokens_per_sec = BATCH_SIZE × OUTPUT_LEN / avg_latency_seconds
```
e.g. 8 × 32 / 4.24 s ≈ 60 tok/s

**Never compare latency_ms across runs with different BATCH_SIZE.** When batch size changes, `avg_latency_ms` will always increase proportionally — only `tokens_per_sec` is the valid cross-batch-size metric. `bench_compare.py` handles this automatically.

---

## Phase 0 — Environment + Branch Setup

Run once at the start of every session.

```bash
# 1. Install uv if missing (needed for venv management)
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Create venv with Python 3.12 if it doesn't exist
if [ ! -f .venv/bin/python ]; then
  uv venv --python 3.12
fi

# 3. Install vllm (CPU build — no precompiled wheel exists for macOS arm64)
# VLLM_USE_PRECOMPILED=1 does NOT work on macOS — only Linux wheels exist.
# This build takes ~4 minutes. Skip if vllm already importable.
.venv/bin/python -c "import vllm" 2>/dev/null || \
  VLLM_TARGET_DEVICE=cpu uv pip install -e . --torch-backend=cpu

# 4. Verify
.venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
.venv/bin/python -c "import torch; print('torch', torch.__version__)"

# 5. Pre-download model weights (skip if already cached in ~/.cache/huggingface)
.venv/bin/python -c "
import os, glob
cache = os.path.expanduser('~/.cache/huggingface/hub')
if glob.glob(f'{cache}/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'):
    print('Model already cached.')
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    print('Downloaded.')
"

# 6. Check available RAM and set GPU_MEM_UTIL accordingly
# On Apple Silicon 16GB with ~5GB free, 0.3 is safe. Adjust if more RAM is free.
python3 -c "
import subprocess, json
mem_bytes = int(subprocess.check_output(['sysctl','-n','hw.memsize']).strip())
mem_gb = mem_bytes / 1024**3
# Use at most 40% of total, leaving room for OS + model weights
util = min(0.4, 4.0 / mem_gb)
print(f'Suggested --gpu-memory-utilization {util:.2f}  (RAM: {mem_gb:.0f}GB)')
"
# Default safe value: --gpu-memory-utilization 0.3

# 7. Create the optimization branch (all code changes land here)
git checkout -b perf/vllm-cpu-opt 2>/dev/null || git checkout perf/vllm-cpu-opt

# 8. Create results dir and visibility log (workspace-relative, not /tmp)
mkdir -p perf_results docs/perf
cat > docs/perf/cpu_opt_log.md << 'EOF'
# vLLM CPU Optimization Log

**Model:** Qwen/Qwen2.5-0.5B-Instruct | **Device:** cpu | **batch=8** | **in=32 out=32**

| # | Fix | tok/s | Delta | Commit |
|---|-----|-------|-------|--------|
EOF
```

---

## Phase 1 — Baseline Benchmark

Run the latency benchmark and save a self-describing JSON via `bench_compare.py --wrap`.

```bash
# Step 1a: benchmark pass (saves raw JSON)
# IMPORTANT: use `.venv/bin/vllm bench latency`, NOT `python -m vllm.benchmarks.latency`
# (the module has no __main__ block — it silently exits with code 0 and produces nothing)
.venv/bin/vllm bench latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.3 \
  --batch-size 8 \
  --input-len 32 \
  --output-len 32 \
  --num-iters-warmup 2 \
  --num-iters 5 \
  --output-json perf_results/baseline_raw.json

# Step 1b: enrich with metadata and compute tokens/sec
.venv/bin/python .cursor/skills/vllm-perf/bench_compare.py \
  --wrap perf_results/baseline_raw.json \
  --batch-size 8 \
  --output-len 32 \
  --label "baseline: bfloat16, batch=8, in=32, out=32" \
  --save perf_results/baseline.json \
  --append-log perf_results/run_log.jsonl
```

After Step 1b, read the printed `tokens_per_sec` value. This is your baseline.

Append to the log table:
```
| 0 | Baseline (bfloat16, batch=8) | {tok/s} | — | N/A |
```

Set the current best: `BEST_JSON=/tmp/vllm_perf/baseline.json`

---

## Phase 2 — Hypothesis Generation (the AI reasoning step)

**This is the core intelligence step. Run it before every iteration — including the first one.**

You are not following a script here. You are an AI agent with access to code, profiling data, and run history. Your job is to reason from evidence and propose the single most impactful untried fix.

### Step 2a — Gather Evidence

Read the following before forming any hypothesis:

```bash
# 1. Current run history (what has been tried, what worked, what didn't)
cat /tmp/vllm_perf/run_log.jsonl 2>/dev/null || echo "(no runs yet — this is iteration 1)"

# 2. Profiler summary (if a profile has been captured)
cat /tmp/vllm_perf/prof_summary.txt 2>/dev/null || echo "(no profile yet)"
```

Also read (skim for inefficiencies, redundant work, or untuned defaults):
- `vllm/v1/worker/cpu_model_runner.py` — forward pass, input batch construction, KV cache write ops
- `vllm/v1/core/sched/scheduler.py` — batching decisions, scheduling overhead
- `vllm/config/scheduler.py` — which knobs exist and what their defaults are
- `vllm/platforms/cpu.py` — what the CPU platform declares about capabilities

### Step 2b — Reason Explicitly

Before proposing anything, answer these questions out loud in your response:

1. **What does the current tokens/sec tell us?**
   - Very low for model size → Python overhead or weight loading per step dominates
   - Scales linearly with batch_size → compute underutilized, increase batch
   - Plateaus quickly with batch_size → memory bandwidth is the ceiling
   - Flag fixes did nothing → bottleneck is in code, not config

2. **What does the profiler show (if captured)?**
   - Which op dominates CPU time? matmul / attention / copy / Python frames?
   - Are there unexpected ops taking significant time?

3. **What have previous iterations revealed?**
   - Which fixes worked? What bottleneck did they relieve?
   - Which fixes failed? What does that eliminate as a hypothesis?

4. **What does the source code reveal?**
   - Any obvious inefficiencies in the CPU path?
   - Any config knobs relevant to your workload shape that haven't been tried?
   - Any redundant copies, unnecessary Python loops, or dead branches in the hot path?

### Step 2c — Generate Ranked Hypotheses

Produce a list of 3–5 specific, mechanistically grounded hypotheses. Each must include:

```
Hypothesis N: <short name>
  Mechanism : WHY this would increase tokens/sec — which bottleneck it targets
  Evidence  : what in the profiler / code / run log supports this
  Fix       : exact flag OR specific code change with file:line
  Est. gain : rough % estimate and confidence: low / medium / high
  Risk      : what could go wrong or make it worse
```

Example of correct hypothesis reasoning:
```
Hypothesis 1: Pin OMP threads to physical (performance) cores
  Mechanism : aten::mm uses OpenBLAS/MKL which parallelizes over OMP threads.
              On Apple M-series, logical core count includes efficiency cores.
              Pinning to physical (P-core) count avoids scheduling on E-cores
              which have lower SIMD throughput.
  Evidence  : OMP_NUM_THREADS not yet set. aten::mm will dominate a 0.5B
              model with 128-token sequences — it is the largest op by FLOP.
  Fix       : OMP_NUM_THREADS=$(sysctl -n hw.physicalcpu) env prefix
  Est. gain : 10–25%, high confidence
  Risk      : Low — env var only, trivially reverted

Hypothesis 2: bitsandbytes INT8 weight quantization
  Mechanism : Reduces every weight tensor from bfloat16 (2 bytes) to int8
              (1 byte), halving the DRAM→cache bandwidth required for each
              aten::mm. On a memory-bandwidth-bound CPU path this directly
              translates to ~2× throughput.
  Evidence  : 0.5B model with 128-token batch is almost certainly
              bandwidth-bound (model fits in L3 cache but weights must be
              streamed per layer per token).
  Fix       : --quantization bitsandbytes --load-format bitsandbytes
  Est. gain : 40–80%, medium confidence (bitsandbytes CPU support required)
  Risk      : Medium — verify bitsandbytes CPU kernels are available;
              run a smoke test before full benchmark
```

### Step 2d — Select and Justify

Pick the **highest-ranked untried hypothesis** (cross-check `/tmp/vllm_perf/run_log.jsonl`). State your selection explicitly before proceeding:

```
SELECTED: Hypothesis N — <name>
REASON  : <one sentence grounded in the evidence above>
NEXT    : proceed to Phase 3 with this specific fix
```

### Fallback Seed Catalog (use only if reasoning produces no better idea)

If this is iteration 1 and no profiling data or code inspection has surfaced a better hypothesis, use this ordered list as seeds:

| Priority | Fix | Flags |
|---|---|---|
| 1 | OMP thread pinning | `OMP_NUM_THREADS=$(sysctl -n hw.physicalcpu)` |
| 2 | Batch size × 2 | `--batch-size 16` |
| 3 | Batch size × 4 | `--batch-size 32` |
| 4 | torch.compile level 3 | `--compilation-config '{"level":3}'` |
| 5 | bitsandbytes INT8 | `--quantization bitsandbytes --load-format bitsandbytes` |
| 6 | Block size 16 | `--block-size 16` |
| 7 | Chunked prefill off | `--no-enable-chunked-prefill` |

**Stop iterating when any of these is true:**
- `MAX_ITERS` (5) attempts completed
- 2 consecutive attempts gained < 3%
- All hypotheses exhausted and no new ones can be formed from evidence

If stopped but tokens/sec still unsatisfying → proceed to Phase 4 (profiling) to gather the evidence needed for the next ideation round.

---

## Phase 3 — Run, Validate, Commit or Revert

Replace `N` with the current iteration number (1, 2, 3…).
Replace `<extra-flags>` with the flags for the current catalog entry.

### Step 3a — Run candidate benchmark

```bash
# Use .venv/bin/vllm bench latency (NOT python -m vllm.benchmarks.latency)
# Always include --gpu-memory-utilization 0.3 (required on Apple Silicon 16GB)
# When changing --batch-size, bench_compare.py will automatically skip latency
# comparison and only evaluate tokens_per_sec (the only valid cross-batch metric)
.venv/bin/vllm bench latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.3 \
  --batch-size 8 \
  --input-len 32 \
  --output-len 32 \
  --num-iters-warmup 2 \
  --num-iters 5 \
  <extra-flags> \
  --output-json perf_results/candidate_N_raw.json

.venv/bin/python .cursor/skills/vllm-perf/bench_compare.py \
  --wrap perf_results/candidate_N_raw.json \
  --batch-size <N> \
  --output-len 32 \
  --label "<short fix description>" \
  --save perf_results/candidate_N.json \
  --append-log perf_results/run_log.jsonl
```

### Step 3b — Compare

```bash
.venv/bin/python .cursor/skills/vllm-perf/bench_compare.py \
  --baseline perf_results/baseline.json \
  --candidate perf_results/candidate_N.json
```

### Step 3c — Accept (exit code 0) or Reject (exit code 1)

**If accepted (tokens/sec improved ≥ 3%, no metric degraded > 5%):**

```bash
# For flag-only fixes: record winning flags
echo "export VLLM_PERF_FLAGS='<all accumulated winning flags>'" \
  >> .cursor/skills/vllm-perf/winning_config.sh

# For file edits (Tier 2/3 only):
git add -A
git commit -m "perf(cpu): <one-line fix description>

tokens/sec: {baseline_tps} → {candidate_tps} (+{delta}%)
batch_size=8, input_len=128, output_len=128

Co-authored-by: Claude"

COMMIT_SHA=$(git rev-parse --short HEAD)

# Update best baseline for next iteration
cp /tmp/vllm_perf/candidate_N.json /tmp/vllm_perf/baseline.json

# Append win to log (use N/A for commit if flag-only)
# Edit docs/perf/cpu_opt_log.md: append table row
# | N | <fix description> | {candidate_tps} | +{delta}% | {COMMIT_SHA or N/A} |
```

**If rejected (exit code 1):**

```bash
# Revert file changes if any were made
git checkout -- .

# Append miss to log
# | N | <fix description> | — | {delta}% (miss) | reverted |
```

After logging, increment N and **return to Phase 2 (Hypothesis Generation)** — re-read the updated run log and reason about what the new result reveals before picking the next fix.

---

## Phase 4 — Profile (only after Phase 2 stop condition)

Profiling is a **separate** invocation. `--profile` does not save `--output-json`.

```bash
# Profile pass only — no JSON output
.venv/bin/python -m vllm.benchmarks.latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu \
  --dtype bfloat16 \
  --batch-size 8 \
  --input-len 128 \
  --output-len 128 \
  --num-iters-warmup 2 \
  --num-iters 1 \
  --profile \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "/tmp/vllm_perf/prof",
    "warmup_iterations": 0,
    "active_iterations": 1
  }'

# Print layerwise summary
.venv/bin/python tools/profiler/print_layerwise_table.py \
  --output-file /tmp/vllm_perf/prof --type summary
```

Save the profiler summary to a file so Phase 2 ideation can read it:
```bash
.venv/bin/python tools/profiler/print_layerwise_table.py \
  --output-file /tmp/vllm_perf/prof --type summary \
  > /tmp/vllm_perf/prof_summary.txt 2>&1
cat /tmp/vllm_perf/prof_summary.txt
```

**Do not pick the next fix here.** The profiler output is evidence. Return to **Phase 2 (Hypothesis Generation)** and let the ideation step reason about what the profiler shows — the bottleneck table below is only a rough guide, not a decision tree:

| Top CPU time consumer | Likely bottleneck | Hypothesis seeds |
|---|---|---|
| `aten::mm` / `aten::linear` > 60% | Weight matmul, memory-bandwidth bound | bitsandbytes INT8, AWQ cpu_wna16, larger batch |
| `aten::scaled_dot_product_attention` > 30% | Attention bandwidth (long seqs) | `--block-size` sweep, chunked prefill off |
| Python frames (`schedule`, `step`) > 10% wall | Scheduler Python overhead | Code optimization in `scheduler.py`, async scheduling |
| `aten::copy_` > 15% | KV cache writeback | `VLLM_CPU_KVCACHE_SPACE` env var, block size |

The agent must reason about the specific numbers, not pattern-match to the table.

---

## Phase 5 — Final Report

When the loop ends, commit the log file and summarize:

```bash
git add docs/perf/cpu_opt_log.md
git commit -m "perf(cpu): add optimization run log

Co-authored-by: Claude"
```

Report to the user:
- Starting tokens/sec (baseline)
- Final tokens/sec (best)
- Total improvement %
- Which fixes worked and which did not
- Link to `docs/perf/cpu_opt_log.md` for full history

---

## Revert Any Change

Every accepted file-edit fix is a commit on `perf/vllm-cpu-opt`.

```bash
# Revert a specific commit (non-destructive):
git revert <SHA>

# Hard-reset to before the optimization session:
git checkout main   # or whichever branch you started from
git branch -D perf/vllm-cpu-opt
```

Flag-only wins are recorded in `.cursor/skills/vllm-perf/winning_config.sh` — simply remove the unwanted line to drop that flag.

---

## Key Source Files

- [`vllm/benchmarks/latency.py`](../../vllm/benchmarks/latency.py) — `--batch-size`, `--output-json`, `--profile` flags
- [`vllm/v1/worker/cpu_model_runner.py`](../../vllm/v1/worker/cpu_model_runner.py) — CPU forward pass
- [`vllm/platforms/cpu.py`](../../vllm/platforms/cpu.py) — CPU platform detection
- [`vllm/config/scheduler.py`](../../vllm/config/scheduler.py) — scheduler knobs
- [`tools/profiler/print_layerwise_table.py`](../../tools/profiler/print_layerwise_table.py) — trace analysis
- [`vllm/model_executor/layers/quantization/cpu_wna16.py`](../../vllm/model_executor/layers/quantization/cpu_wna16.py) — AWQ CPU quant
- [`.cursor/skills/vllm-perf/bench_compare.py`](bench_compare.py) — comparison harness (this skill's helper)
