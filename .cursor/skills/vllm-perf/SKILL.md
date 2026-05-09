# vLLM CPU Performance Optimization Skill

## When to Use This Skill

Read and follow this skill when the user asks to:
- "make vLLM faster", "optimize tokens per second", "run the perf loop", "benchmark inference"
- improve throughput on Apple Silicon / macOS

**Read this entire file before executing any step.**

---

## What This Skill Does

This skill runs an autonomous optimization loop that alternates between two fix types:

- **Flag/Config fixes** — CLI flags, environment variables, scheduler knobs
- **Code fixes** — targeted edits to vLLM Python source files

Code fixes are **not optional add-ons**. Once cheap flags are exhausted, the loop must pivot to reading source code, identifying waste in the hot path, patching it, smoke-testing correctness, and benchmarking. Both fix types follow the same measure-commit-or-revert discipline.

---

## Platform Constraints

- vLLM has **no MPS backend** on macOS. The only valid device is CPU.
- All optimizations in this skill are CPU-compatible only.
- **`--device cpu` is NOT a valid flag for `vllm bench latency`** — it will error. Device is auto-detected. `--device cpu` only works with `python -m vllm.benchmarks.latency` (legacy module path used only for profiling).
- Benchmarking (`--output-json`) and profiling (`--profile`) are **mutually exclusive** in one invocation — always run them as separate commands.
- **"Inductor compilation was disabled by user settings" warning is a false alarm on CPU.** The CPU platform's `inference_mode()` converts mode=VLLM_COMPILE → DYNAMO_TRACE_ONCE + inductor backend. The warning fires in a second VllmConfig `__post_init__` (after the platform has already set up compilation correctly) because mode≠VLLM_COMPILE at that point. The model IS compiled with torch.compile + inductor. Do not treat this warning as a hypothesis.

### ARM CPU dtype constraint (CRITICAL)

**Apple M1/M2/M3/M4 AMX units have NO native bfloat16 compute.** PyTorch falls back
to scalar emulation for `aten::mm` with bf16 inputs on ARM, making matmul **~632×
slower** than float32. This was confirmed via micro-benchmark (512×512 matmul:
bf16 = 44.5s vs fp32 = 0.07s, same hardware).

**Always use `--dtype float32` on ARM CPU.** Never use `--dtype bfloat16` or
`--dtype auto` (which picks the model's native dtype, typically bf16 for modern models).
On x86 with AVX-512 BF16 extension, bfloat16 is fine.

---

## Constants (do not change without user instruction)

```
MODEL              = Qwen/Qwen2.5-0.5B-Instruct
DTYPE              = float32   # CRITICAL on ARM — see Platform Constraints
BATCH_SIZE         = 8         # starting point; may grow as optimizations accumulate
INPUT_LEN          = 32        # keeps each benchmark iter under 2 min on CPU
OUTPUT_LEN         = 32        # same reason
GPU_MEM_UTIL       = 0.3       # CRITICAL: default 0.92 OOMs on 16 GB Mac with ~5 GB free
MAX_ITERS          = 5
RESULTS_DIR        = perf_results   # workspace-relative — survives reboots
BRANCH             = perf/vllm-cpu-opt
LOG_FILE           = docs/perf/cpu_opt_log.md
SKILL_DIR          = .cursor/skills/vllm-perf
```

**tokens/sec formula (the only valid cross-run metric):**
```
tokens_per_sec = BATCH_SIZE × OUTPUT_LEN / avg_latency_seconds
```
e.g. 8 × 32 / 4.24 s ≈ 60 tok/s

Never compare `avg_latency_ms` across runs with different `BATCH_SIZE` or `INPUT_LEN`.
Only `tokens_per_sec` is valid cross-batch. `bench_compare.py` enforces this automatically
when batch_size differs, but does NOT automatically detect input_len mismatches.

---

## Phase 0 — Environment + Branch Setup

Run once at the start of every session.

```bash
# 1. uv
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. venv
[ -f .venv/bin/python ] || uv venv --python 3.12

# 3. Install vllm CPU build (skip if already importable)
# VLLM_USE_PRECOMPILED=1 does NOT work on macOS — no arm64 wheels exist.
# Build takes ~4 min.
.venv/bin/python -c "import vllm" 2>/dev/null || \
  VLLM_TARGET_DEVICE=cpu uv pip install -e . --torch-backend=cpu

# 4. Verify
.venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
.venv/bin/python -c "import torch; print('torch', torch.__version__)"

# 5. Pre-download model weights (skip if cached)
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

# 6. Check RAM + CPU arch
.venv/bin/python -c "
import subprocess, platform
mem_gb = int(subprocess.check_output(['sysctl','-n','hw.memsize']).strip()) / 1024**3
util = min(0.4, 4.0 / mem_gb)
arch = platform.machine()
print(f'CPU: {arch}, RAM: {mem_gb:.0f} GB')
print(f'Suggested --gpu-memory-utilization {util:.2f}')
if arch == 'arm64':
    print('ARM detected → MUST use --dtype float32 (bf16 has no AMX support)')
else:
    print('x86 detected → bf16 OK if AVX-512 BF16 extension present')
"

# 7. Optimization branch (all changes land here)
git checkout -b perf/vllm-cpu-opt 2>/dev/null || git checkout perf/vllm-cpu-opt

# 8. Results dir + log
mkdir -p perf_results docs/perf
[ -f docs/perf/cpu_opt_log.md ] || cat > docs/perf/cpu_opt_log.md << 'EOF'
# vLLM CPU Optimization Log

**Model:** Qwen/Qwen2.5-0.5B-Instruct | **Device:** cpu | **in=32 out=32**

| # | Type | Fix | tok/s | Delta | Commit |
|---|------|-----|-------|-------|--------|
EOF

# 9. Validate existing baseline.json before skipping Phase 1.
#    The raw JSON from `vllm bench latency` does NOT store input_len.
#    bench_compare.py --wrap trusts whatever --input-len you pass on the CLI.
.venv/bin/python -c "
import json, sys, os
path = 'perf_results/baseline.json'
if not os.path.exists(path):
    print('No baseline.json — must run Phase 1.')
    sys.exit(0)
b = json.load(open(path))
stored_in  = b.get('input_len')
stored_out = b.get('output_len')
stored_bs  = b.get('batch_size')
tps  = b.get('tokens_per_sec', 0)
lat  = b.get('avg_latency_s', 0)
computed_bs_x_out = tps * lat
expected = stored_bs * stored_out if stored_bs and stored_out else 0
print(f'Baseline: batch={stored_bs}, input_len={stored_in}, output_len={stored_out}')
print(f'  tokens_per_sec = {tps:.1f}, avg_latency = {lat:.2f}s')
print(f'  Implied batch*output = {computed_bs_x_out:.0f} (expected {expected})')
if expected and abs(computed_bs_x_out - expected) > expected * 0.05:
    print('WARNING: Mismatch >5%. Baseline may be mislabeled. Re-run Phase 1.')
else:
    print('Baseline integrity: OK')
" 2>/dev/null || echo "baseline.json missing or unreadable — run Phase 1"
```

---

## Phase 0.5 — Platform Micro-Benchmarks (run once per machine)

**Run this before Phase 1 if no prior session has validated dtype performance.**
This takes <60 seconds and catches pathological configs before burning benchmark time.

```bash
.venv/bin/python -c "
import torch, time, platform

arch = platform.machine()
print(f'Architecture: {arch}')

for dtype_name, dtype in [('float32', torch.float32), ('bfloat16', torch.bfloat16)]:
    a = torch.randn(512, 512, dtype=dtype)
    b = torch.randn(512, 512, dtype=dtype)
    # warmup
    for _ in range(5):
        torch.mm(a, b)
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        torch.mm(a, b)
    elapsed = time.perf_counter() - t0
    print(f'  {dtype_name} 512x512 matmul x{n}: {elapsed*1000:.1f} ms ({elapsed/n*1000:.2f} ms/op)')

print()
print('DECISION: If bf16 is >5x slower than fp32, use --dtype float32.')
print('          If roughly equal, bf16 saves memory — prefer --dtype bfloat16.')
"
```

**Expected results by architecture:**
- **ARM (Apple M1/M2/M3/M4):** bf16 ~600× slower. Use `--dtype float32`.
- **x86 with AVX-512 BF16:** bf16 ≈ fp32 or faster. Use `--dtype bfloat16`.
- **x86 without BF16 extension:** bf16 ~2-10× slower. Use `--dtype float32`.

---

## Phase 1 — Baseline Benchmark

**Skip this phase only if baseline.json exists AND the Phase 0 sanity check
confirmed batch×output matches (i.e., `implied batch×output ≈ stored batch×output`).**

If baseline.json exists but fails the sanity check, overwrite it by re-running this phase
with the correct parameters. Never carry forward a stale or mislabeled baseline.

**Use the dtype determined by Phase 0.5.** On ARM, this MUST be float32.

```bash
# Step 1a: benchmark
# Use .venv/bin/vllm bench latency — NOT python -m (no __main__, produces nothing)
.venv/bin/vllm bench latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float32 \
  --gpu-memory-utilization 0.3 \
  --batch-size 8 \
  --input-len 32 \
  --output-len 32 \
  --num-iters-warmup 2 \
  --num-iters 5 \
  --output-json perf_results/baseline_raw.json

# Step 1b: enrich (always pass --input-len and --output-len — raw JSON omits both)
.venv/bin/python .cursor/skills/vllm-perf/bench_compare.py \
  --wrap perf_results/baseline_raw.json \
  --batch-size 8 \
  --input-len 32 \
  --output-len 32 \
  --label "baseline: float32, batch=8, in=32, out=32" \
  --save perf_results/baseline.json \
  --append-log perf_results/run_log.jsonl
```

Read the printed `tokens_per_sec`. Append to log:
```
| 0 | flag | Baseline (float32, batch=8) | {tok/s} | — | N/A |
```

**Expected runtime:** ~30s for warmup + 5 iters at batch=8 with float32 on M1.
If this takes >2 minutes, something is wrong — likely wrong dtype. Stop and investigate.

---

## Phase 2 — Hypothesis Generation

**This is the core intelligence step. Run it before EVERY iteration.**

You are an AI agent with code access, profiling data, and run history. Your job is to reason from evidence and pick the single highest-impact untried fix — flag or code.

### Step 2a — Read the Evidence

```bash
# Run history
cat perf_results/run_log.jsonl 2>/dev/null || echo "(no runs yet)"

# Profiler summary (if captured)
cat perf_results/prof_summary.txt 2>/dev/null || echo "(no profile yet)"
```

### Step 2b — Code Scan (mandatory from iteration 3 onward, or whenever flag fixes stall)

Read these files looking for hot-path waste. Skim for the anti-patterns listed below.

**Files to read:**
- `vllm/v1/worker/gpu_model_runner.py` — the inherited forward loop CPU runs through
- `vllm/v1/worker/cpu_model_runner.py` — CPU-specific overrides
- `vllm/v1/utils.py` — `CpuGpuBuffer` definition
- `vllm/v1/core/sched/scheduler.py` — per-step scheduling overhead
- `vllm/platforms/cpu.py` — CPU platform knobs and constraints

**Anti-patterns to look for (with search commands):**

```bash
# 1. Self-copies: copy_to_gpu() after _postprocess_tensors sets .gpu = .cpu
#    NOTE: Empirically tested (2026-05): gain was only +0.5% (noise floor).
#    PyTorch CPU copy_() on self is internally near-free. Skip this hypothesis
#    unless profiler shows aten::copy_ > 15% of wall time.
grep -n "copy_to_gpu\|copy_to_cpu" vllm/v1/worker/gpu_model_runner.py

# 2. Per-step numpy allocations: np.repeat / np.cumsum / torch.from_numpy
#    During pure decode (all batch=N, output 1 token each), many of these arrays
#    are identical step-to-step and could be cached or avoided.
grep -n "np\.repeat\|np\.cumsum\|torch\.from_numpy\|np\.array(" \
  vllm/v1/worker/gpu_model_runner.py

# 3. Per-step Python loops over requests
grep -n "for req\|for i in range(num_reqs\|for req_id in" \
  vllm/v1/worker/gpu_model_runner.py | head -30

# 4. Unnecessary .to(device) on tensors already on the right device
grep -n "\.to(self\.device\|\.to(\"cpu\"\|\.to(device" \
  vllm/v1/worker/gpu_model_runner.py | head -30

# 5. Dead branches guarded by flags already False on CPU
grep -n "use_cuda_graph\|cascade_attn" vllm/v1/worker/gpu_model_runner.py | head -20

# 6. Redundant .contiguous() or .clone() in the forward path
grep -n "\.contiguous()\|\.clone()" vllm/v1/worker/gpu_model_runner.py | head -20
```

For any hit, read ±20 lines of context to understand whether it's in the hot path (called every step) and whether it's safe to change.

### Step 2c — Reason Explicitly

Answer these questions out loud before proposing hypotheses:

1. **Is the baseline configuration sane?**
   - Correct dtype for this CPU architecture? (fp32 on ARM, bf16 OK on x86+AVX-512 BF16)
   - Reasonable batch size? (CPU can usually handle larger batches than GPU for small models)
   - Any red flags in the benchmark output? (unexpectedly slow warmup, OOM warnings)

2. **What does current tokens/sec tell us?**
   - Scales with batch (sublinear latency): compute underutilized, try larger batch
   - Plateaus with batch: memory bandwidth is the ceiling, try quantization
   - Flag fixes had zero effect: bottleneck is in code, not config

3. **What does the profiler show (if captured)?**
   - Which op takes the most wall time?
   - Are there surprising ops (unexpected copies, fallback kernels)?

4. **What did previous iterations reveal?**
   - What bottleneck did each accepted fix relieve?
   - What does each rejected fix eliminate?

5. **What did the code scan surface?**
   - Are there per-step allocations, or dead branches in the hot path?
   - Which is cheapest to fix safely?

### Step 2d — Generate Ranked Hypotheses

Produce 3–5 hypotheses. Each must specify its type and include:

```
Hypothesis N: <name>
  Type      : Flag | EnvVar | CodeEdit
  Mechanism : WHY this increases tokens/sec — which bottleneck it removes
  Evidence  : what in profiler / code scan / run log supports this
  Fix       : exact flag/env-var OR file:line with specific change described
  Est. gain : % estimate + confidence: low / medium / high
  Risk      : what could break; how to verify correctness
```

### Step 2e — Select

```
SELECTED: Hypothesis N — <name>
TYPE    : Flag | EnvVar | CodeEdit
REASON  : <one sentence from evidence above>
```

### Fallback Seed Catalog

Use only when no code scan or profiler has surfaced a better hypothesis.

| Priority | Type | Fix | Notes |
|---|---|---|---|
| 1 | Flag | `--batch-size 16` | CPU usually underutilized at small batch |
| 2 | Flag | `--batch-size 32` | Continue if batch×2 showed sublinear latency scaling |
| 3 | Flag | `--batch-size 64` | Continue if still sublinear; watch for OOM |
| 4 | Flag | `--batch-size 128` | Try if 64 still sublinear; diminishing returns likely |
| 5 | CodeEdit | Cache decode-step numpy arrays | `np.repeat`/`np.cumsum` in `_prepare_inputs` rebuild identical arrays every decode step |
| 6 | CodeEdit | Hoist dead `if self.use_cuda_graph:` branches | Always False on CPU |
| 7 | Flag | `--no-enable-chunked-prefill` | Marginal gain at in=32; only try after code fixes stall |

**Disproven / Do Not Use:**

| Fix | Result | Why |
|---|---|---|
| `--dtype bfloat16` on ARM | **632× slower matmul** | M1/M2/M3/M4 AMX has no native bf16 compute |
| `copy_to_gpu()` identity guard | +0.5% (noise) | PyTorch CPU `copy_(self)` is internally near-free |
| `--compilation-config '{"level":3}'` | Errors | `CompilationConfig` has no `level` field; CPU platform already enables torch.compile+inductor |
| `OMP_NUM_THREADS=$(sysctl -n hw.physicalcpu)` | -2.3% | Hurts on M1 Pro; default threading better |
| `--block-size 32` | -0.4% | ARM NEON optimal at default 16 (128-bit = 8×bf16) |

**Stop condition (any one):**
- `MAX_ITERS` (5) attempts completed
- 2 consecutive iterations gained < 3%
- All hypotheses exhausted

If stopped and tokens/sec is still unsatisfying → run Phase 4 (profiling) to gather new evidence, then return to Phase 2.

---

## Phase 3 — Implement, Smoke Test, Benchmark, Commit or Revert

Replace `N` with the current iteration number.

### Step 3a — Implement

**For Flag/EnvVar fixes:** no file changes. Note the flags to add in Step 3b.

**For CodeEdit fixes:**

1. Read the target function in full before editing.
2. Make the minimal targeted change — do not refactor, rename, or touch unrelated lines.
3. Verify the edit with `git diff`.

```bash
git diff vllm/   # confirm only intended lines changed
```

### Step 3b — Smoke Test (mandatory for CodeEdit; recommended for Flag)

Run a single iteration with no output-json to verify the model still produces output and does not crash:

```bash
.venv/bin/vllm bench latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float32 \
  --gpu-memory-utilization 0.3 \
  --batch-size 8 \
  --input-len 32 \
  --output-len 32 \
  --num-iters-warmup 0 \
  --num-iters 1 \
  <extra-flags-if-flag-fix>
```

**Sanity-check the smoke-test latency against the baseline:**
- `tokens/sec ≈ batch × output_len / latency_seconds`
- If the smoke-test latency is **2× or more** than the baseline latency with the same batch+input+output, something is wrong (wrong input_len, wrong dtype, extra system load, OOM swap) — stop and investigate before running the full benchmark.

If the smoke test crashes or produces a traceback → revert immediately, mark as miss:
```bash
git checkout -- .
# | N | code | <fix> | — | crash (miss) | reverted |
```

### Step 3c — Benchmark

Use the **same `--dtype`, `--batch-size`, `--input-len`, and `--output-len`** as the
current `baseline.json`. If you changed batch size or dtype as the candidate fix, that's
fine — but record the change explicitly.

**Expected runtimes (float32, M1 Pro):**
- batch=8:  ~10s for warmup+5iters
- batch=32: ~40s
- batch=64: ~30s (yes, faster than batch=32 at bf16; that's the fp32 speedup)
- batch=128: ~60s

If a run takes >3× the expected time, stop and investigate — likely wrong dtype.

```bash
.venv/bin/vllm bench latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float32 \
  --gpu-memory-utilization 0.3 \
  --batch-size <BATCH_SIZE_USED> \
  --input-len 32 \
  --output-len 32 \
  --num-iters-warmup 2 \
  --num-iters 5 \
  <extra-flags-if-flag-fix> \
  --output-json perf_results/candidate_N_raw.json

# Always pass --input-len and --output-len explicitly — raw JSON omits both.
.venv/bin/python .cursor/skills/vllm-perf/bench_compare.py \
  --wrap perf_results/candidate_N_raw.json \
  --batch-size <BATCH_SIZE_USED> \
  --input-len 32 \
  --output-len 32 \
  --label "<type>: <short fix description>" \
  --save perf_results/candidate_N.json \
  --append-log perf_results/run_log.jsonl
```

### Step 3d — Compare

```bash
.venv/bin/python .cursor/skills/vllm-perf/bench_compare.py \
  --baseline perf_results/baseline.json \
  --candidate perf_results/candidate_N.json
```

`bench_compare.py` exits 0 (accepted) if tokens/sec improved ≥ 3% and no latency
metric regressed > 5%. It exits 1 (rejected) otherwise.

**If batch_size changed:** bench_compare.py will warn "latency metrics not comparable"
and evaluate tokens_per_sec only. This is correct behavior — accept it.

### Step 3e — Accept or Revert

**If accepted (exit code 0):**

```bash
# Record flag-only wins
[ "<type>" = "flag" ] && \
  echo "export VLLM_PERF_FLAGS='<accumulated winning flags>'" \
    >> .cursor/skills/vllm-perf/winning_config.sh

# Commit everything (results + code edits)
git add perf_results/ docs/perf/
# For CodeEdit, also stage the changed source file(s):
# git add vllm/path/to/changed_file.py
git commit -m "perf(cpu): <one-line fix description>

type: <Flag|CodeEdit>
tokens/sec: {baseline_tps} → {candidate_tps} (+{delta}%)
batch_size={N}, input_len=32, output_len=32"

COMMIT_SHA=$(git rev-parse --short HEAD)

# Promote candidate to new baseline
cp perf_results/candidate_N.json perf_results/baseline.json

# Append to log
# | N | <Flag/Code> | <fix description> | {tok/s} | +{delta}% | {SHA or N/A} |
```

**If rejected (exit code 1):**

```bash
# Revert code edits (flag fixes need no revert)
git checkout -- vllm/

# Append miss
# | N | <Flag/Code> | <fix description> | — | {delta}% (miss) | reverted |
```

**After logging:** increment N, return to Phase 2. Re-read `perf_results/run_log.jsonl`
and reason about what the result reveals before picking the next fix.

---

## Phase 4 — Profile

Run when the optimization loop stalls (2 consecutive misses or MAX_ITERS hit) to gather
new evidence for Phase 2 hypothesis generation.

Profiling is a **separate** invocation — `--profile` does not save `--output-json`.
Use `perf_results/prof/` (workspace-relative, survives reboots).

**Use the same dtype as your current baseline (float32 on ARM).**

```bash
mkdir -p perf_results/prof

# NOTE: --device cpu IS valid here (python -m path, not vllm bench latency)
.venv/bin/python -m vllm.benchmarks.latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu \
  --dtype float32 \
  --batch-size 8 \
  --input-len 32 \
  --output-len 32 \
  --num-iters-warmup 2 \
  --num-iters 1 \
  --profile \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "perf_results/prof",
    "warmup_iterations": 0,
    "active_iterations": 1
  }'

# Save summary for Phase 2 evidence reading
.venv/bin/python tools/profiler/print_layerwise_table.py \
  --output-file perf_results/prof --type summary \
  > perf_results/prof_summary.txt 2>&1
cat perf_results/prof_summary.txt
```

**Do not pick the next fix here.** Return to Phase 2 and reason from the numbers.

| Top CPU time consumer | Likely bottleneck | Hypothesis seeds |
|---|---|---|
| `aten::mm` / `aten::linear` > 60% | Weight matmul / bandwidth | quantization, larger batch |
| `aten::copy_` > 15% | Tensor copies in hot path | Cache decode arrays, investigate copy sites |
| `aten::scaled_dot_product_attention` > 30% | Attention bandwidth | block-size sweep, chunked prefill off |
| Python frames > 10% wall | Python overhead per step | Hoist dead branches, cache per-step metadata |

---

## Phase 5 — Final Report

```bash
git add docs/perf/cpu_opt_log.md
git commit -m "perf(cpu): add optimization run log"
```

Report to the user:
- Starting tokens/sec (baseline)
- Final tokens/sec (best candidate)
- Total % improvement
- Table of what worked vs what did not, with types (Flag vs CodeEdit)
- Link to `docs/perf/cpu_opt_log.md`

---

## Reverting

```bash
# Revert a specific commit (non-destructive)
git revert <SHA>

# Hard reset to before the session
git checkout main
git branch -D perf/vllm-cpu-opt
```

Flag wins are in `.cursor/skills/vllm-perf/winning_config.sh` — remove the line to drop a flag.

---

## Key Source Files

Read these when scanning for code-level optimizations:

| File | What to look for |
|---|---|
| `vllm/v1/worker/gpu_model_runner.py` | `np.repeat`/`from_numpy` per step, per-request Python loops, `.to(device)` calls |
| `vllm/v1/worker/cpu_model_runner.py` | CPU-specific overrides; check what the parent still runs that may be wasteful |
| `vllm/v1/utils.py` | `CpuGpuBuffer` — `copy_to_gpu()` implementation (empirically cheap on CPU) |
| `vllm/v1/core/sched/scheduler.py` | Per-step Python object creation, list builds, dict lookups in the decode loop |
| `vllm/platforms/cpu.py` | Platform knobs, dtype support, block size defaults, compilation mode setup |
| `vllm/model_executor/layers/quantization/cpu_wna16.py` | AWQ CPU quant availability |
| `tools/profiler/print_layerwise_table.py` | Profiler output analysis |
| `.cursor/skills/vllm-perf/bench_compare.py` | This skill's benchmark harness |

---

## Empirical Results Log (accumulated across sessions)

Record tested hypotheses here so future sessions don't repeat them.

| Date | Fix | Result | Notes |
|---|---|---|---|
| 2026-05-09 | **dtype=float32 (from bf16)** | **+365.6% ✓** | M1 AMX has no native bf16 — scalar fallback 632× slower |
| 2026-05-09 | batch=8 (from 4) | +43.6% ✓ | CPU underutilized at small batch |
| 2026-05-09 | batch=16 (from 8) | +12.8% ✓ | Diminishing returns beginning |
| 2026-05-09 | batch=32 (from 16) | +7.8% ✓ | Still sublinear latency scaling |
| 2026-05-09 | batch=64 (from 32) | +5.4% ✓ | Still sublinear; approaching plateau |
| 2026-05-09 | OMP_NUM_THREADS=8 | -2.3% ✗ | Hurts on M1 Pro (default threading better) |
| 2026-05-09 | --block-size 32 | -0.4% ✗ | ARM NEON optimal at default 16 |
| 2026-05-09 | copy_to_gpu identity guard | +0.5% ✗ | PyTorch CPU copy_(self) is internally near-free |
