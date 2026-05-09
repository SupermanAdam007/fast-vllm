# vLLM CPU Performance Optimization Skill

## When to Use This Skill

Read and follow this skill when the user asks to:
- "make vLLM faster", "optimize tokens per second", "run the perf loop", "benchmark inference"
- improve throughput on Apple Silicon / macOS

**Read this entire file before executing any step.**

---

## What This Skill Does

This skill runs an optimization loop that **improves the vLLM implementation** to squeeze more tokens/second out of the CPU path. There are three fix types, ordered by impact:

1. **CodeEdit** — the primary work. Targeted edits to vLLM source: attention kernels, linear layer dispatch, KV cache operations, operator fusion, quantization hooks. These are permanent improvements that hold at every batch size and compound with future changes.
2. **InductorConfig** — `--compilation-config '{"inductor_compile_config": {...}}'` tweaks to how torch inductor compiles the model's forward pass.
3. **Flag** — CLI flags, env vars, scheduler knobs. **Flag optimization is complete for this setup** (batch=256, float32 is the stable plateau). Do not revisit flags unless a code change invalidates the baseline.

### Current optimization status

Flag/batch tuning is exhausted as of 2026-05-09:
- dtype=float32: mandatory on ARM (bfloat16 is 632× slower due to no AMX support)
- batch=256: plateau (latency ratio 1.91× at 2× batch — memory-bandwidth bound)
- All other flags tested and either regressed or below noise floor

**Sessions should start CodeEdit work directly.** Skip batch tuning entirely unless the profiler shows a new compute regime. The question is no longer "what flags to set" — it is "what in the vLLM implementation is leaving throughput on the table."

### What "compute-bound" means for code targets

At batch=256, float32, ~18s per iteration (32 decode steps), each decode step takes ~550ms. Model compute (matmuls, attention) dominates. This means:

- **Python scheduling overhead** (`_prepare_inputs`, numpy allocations) is ~1% of wall time → **below measurement noise floor**. Do not target this layer. Empirically confirmed 2026-05.
- **The compiled forward pass** (GEMM kernels, attention, KV cache reads) is the actual bottleneck. Code changes here are where gains live.
- **Quantization** reduces weight memory traffic; on a memory-bandwidth-bound regime it directly reduces latency.

Both CodeEdit and InductorConfig fixes follow the measure-commit-or-revert discipline.

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
BATCH_SIZE         = 256       # stable plateau — do NOT change without profiler evidence
INPUT_LEN          = 32        # keeps each benchmark iter ~90s on CPU at batch=256
OUTPUT_LEN         = 32        # same reason
KV_CACHE_SPACE     = 1         # GiB — set via VLLM_CPU_KVCACHE_SPACE=1 env var
                               # Bypasses the gpu_memory_utilization formula entirely.
                               # Formula: kv_cache = total×util − model_rss → goes negative
                               # when model_rss > total×util (float32 model ≈ 3.2 GB in-process).
                               # Use VLLM_CPU_KVCACHE_SPACE=1 in ALL bench commands instead of
                               # --gpu-memory-utilization. 1 GiB >> 32 MB needed at batch=256.
GPU_MEM_UTIL       = 0.3       # DEPRECATED for benchmarking — kept for reference only.
                               # Use KV_CACHE_SPACE above. --gpu-memory-utilization 0.3 fails
                               # whenever system free RAM < 4.8 GB (Cursor + OS + model use ~11+ GB).
MAX_ITERS          = 5
RESULTS_DIR        = perf_results   # workspace-relative — survives reboots
BRANCH             = perf/vllm-cpu-opt
LOG_FILE           = docs/perf/cpu_opt_log.md
SKILL_DIR          = .cursor/skills/vllm-perf
```

**Stable baseline (as of 2026-05-09):** 463.6 tok/s at batch=256, float32, in=32, out=32.
If `perf_results/baseline.json` shows a higher number, use that — it means a prior session landed a code improvement.

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

# 6. Check RAM + CPU arch + memory pressure
.venv/bin/python -c "
import subprocess, platform, os

arch = platform.machine()
total_gb = int(subprocess.check_output(['sysctl','-n','hw.memsize']).strip()) / 1024**3

# macOS available memory: free + inactive pages
vm = subprocess.check_output(['vm_stat']).decode()
page = 4096
free = inactive = 0
for line in vm.splitlines():
    if 'Pages free' in line: free = int(line.split(':')[1].strip().rstrip('.'))
    if 'Pages inactive' in line: inactive = int(line.split(':')[1].strip().rstrip('.'))
avail_gb = (free + inactive) * page / 1024**3

print(f'CPU: {arch}, Total RAM: {total_gb:.0f} GB, Available: {avail_gb:.1f} GB')
if arch == 'arm64':
    print('ARM detected → MUST use --dtype float32 (bf16 has no AMX support)')
else:
    print('x86 detected → bf16 OK if AVX-512 BF16 extension present')

# Memory check for benchmark viability
if avail_gb < 5.0:
    print()
    print('WARNING: Available RAM < 5 GB. --gpu-memory-utilization 0.3 WILL FAIL.')
    print('  The formula (total×util - model_rss) goes negative when model_rss > 3.2 GB.')
    print('  USE: VLLM_CPU_KVCACHE_SPACE=1 (no --gpu-memory-utilization flag)')
    print('  This sets explicit 1 GiB KV cache, bypassing the formula entirely.')
    print('  1 GiB >> 32 MB needed at batch=256 — no throughput impact.')
else:
    print(f'RAM OK: {avail_gb:.1f} GB free — VLLM_CPU_KVCACHE_SPACE=1 still recommended (more robust)')
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
# VLLM_CPU_KVCACHE_SPACE=1 sets explicit 1 GiB KV cache, bypassing the
# gpu_memory_utilization formula that fails when system RAM < 5 GB free.
VLLM_CPU_KVCACHE_SPACE=1 .venv/bin/vllm bench latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float32 \
  --batch-size 256 \
  --input-len 32 \
  --output-len 32 \
  --num-iters-warmup 2 \
  --num-iters 5 \
  --output-json perf_results/baseline_raw.json

# Step 1b: enrich (always pass --input-len and --output-len — raw JSON omits both)
.venv/bin/python .cursor/skills/vllm-perf/bench_compare.py \
  --wrap perf_results/baseline_raw.json \
  --batch-size 256 \
  --input-len 32 \
  --output-len 32 \
  --label "baseline: float32, batch=256, in=32, out=32" \
  --save perf_results/baseline.json \
  --append-log perf_results/run_log.jsonl
```

Read the printed `tokens_per_sec`. Append to log:
```
| 0 | flag | Baseline (float32, batch=256) | {tok/s} | — | N/A |
```

**Expected baseline:** ~463 tok/s. If significantly lower (< 420), check for background load or wrong dtype.
**Expected runtime:** ~90s for warmup + 5 iters at batch=256 with float32 on M1.

---

## Phase 2 — Hypothesis Generation

**This is the core intelligence step. Run it before EVERY iteration.**

You are working on improving the vLLM implementation. Flag tuning is done. Your job is to find code-level changes to the compiled forward pass — attention, linear layers, KV cache operations — that reduce the per-token compute cost.

### Step 2a — Read the Evidence

```bash
# Run history — check what has been tried and what each result ruled out
cat perf_results/run_log.jsonl 2>/dev/null || echo "(no runs yet)"

# Profiler summary — the most valuable input for hypothesis generation
cat perf_results/prof_summary.txt 2>/dev/null || echo "(no profile yet — run Phase 4 before iteration 1)"
```

**If no profiler data exists yet, run Phase 4 first. This is a hard rule, not a suggestion.**

Empirical lesson (2026-05): two sessions skipped the profiler and went straight to catalog hypotheses. Both misses. The catalog items `max_autotune_gemm_backends=AT_BLAS` (+2.2%) and `cpp.simdlen=256` (+0.7%) produced sub-threshold results because we had no profiler evidence that GEMM dispatch was the actual bottleneck. The profiler is the only way to know which op to target. **STOP. Do not proceed to Step 2d until `perf_results/prof_summary.txt` exists and has been read.**

### Step 2b — Forward-Pass Code Scan

Scan the compiled forward path. At batch=256, the bottleneck is in GEMM and attention — not in Python scheduling.

**Primary scan targets:**

```bash
# 1. Attention backend — how decode attention is implemented for CPU
# NOTE: path is vllm/v1/attention/backends/ — NOT vllm/attention/backends/ (doesn't exist)
ls vllm/v1/attention/backends/
grep -rn "class.*Backend\|def forward" vllm/v1/attention/backends/ | grep -v test | head -30

# 2. Which attention backend is actually selected for CPU
grep -n "get_attn_backend\|_attn_backend\|attention_backend\|CPU_ATTN" vllm/platforms/cpu.py
grep -n "CPU_ATTN\|get_attn_backend_cls" vllm/v1/attention/backends/registry.py 2>/dev/null | head -10

# 3. Linear layer dispatch — is the optimal GEMM path selected?
grep -n "def forward\|F\.linear\|torch\.mm\|torch\.matmul\|aten::mm" \
  vllm/model_executor/layers/linear.py | head -30

# 4. KV cache operations — how keys/values are written and read each step
grep -rn "def forward\|cache_k\|cache_v\|kv_cache\|paged_attn" \
  vllm/v1/attention/backends/ | grep -v test | head -30

# 5. Inductor config currently active — what's already enabled
grep -n "inductor_compile_config\|epilogue_fusion\|cpp\." vllm/platforms/cpu.py | head -20

# 6. Quantization availability — is AWQ/INT8 usable on CPU?
cat vllm/model_executor/layers/quantization/cpu_wna16.py 2>/dev/null | head -40
grep -rn "is_cpu\|cpu.*quant\|quant.*cpu" vllm/model_executor/layers/quantization/ \
  --include="*.py" | grep -v test | head -20
```

**For each hit, read ±30 lines of context** to understand: (a) is this in the decode hot path? (b) is there a more efficient alternative that works on CPU/ARM?

### Step 2c — Reason Explicitly

Answer these before proposing hypotheses:

1. **What does the profiler say dominates?** (If no profiler data: run Phase 4 now, before hypothesizing.)
   - `aten::mm` / `aten::linear` > 60%: GEMM-bound → quantization, better kernel dispatch
   - `aten::scaled_dot_product_attention` > 30%: attention-bound → block size, flash attention path
   - `aten::index_put` / `aten::copy_` > 15%: KV cache write bound → layout or fused kernel
   - `aten::index` / `aten::gather` > 10%: KV cache read bound → investigate paged attention path

2. **Which attention backend is active?** Check `vllm/platforms/cpu.py`.
   - Is it `TorchSDPABackend` (generic torch)? A CPU-optimized backend might exist.
   - Is IPEX available? Intel Extension for PyTorch has CPU-specific optimized attention.
   - Is the attention running MHA or GQA? Qwen2.5-0.5B uses GQA (14 heads, 2 KV heads) — verify the backend handles this efficiently.

3. **Is weight quantization viable?**
   - Weight-only INT8 (AWQ/GPTQ) halves the weight memory read per matmul. At batch=256, if `aten::mm` dominates and the model is memory-bandwidth-bound (verified by latency ratio 1.91× at batch doubling), quantization directly reduces the bottleneck.
   - Check `vllm/model_executor/layers/quantization/cpu_wna16.py` — does it support Qwen2.5?

4. **What inductor options haven't been tried?**
   - `cpp.simdlen` and `max_autotune_gemm_backends=AT_BLAS`: **both empirically tested and rejected** (2026-05). AT_BLAS gave +2.2% (below 3% threshold); simdlen=256 gave +0.7%. LLVM already auto-vectorizes ARM NEON well; neither option meaningfully changes the compiled kernels. Do not re-try unless the profiler shows a new kernel pattern.
   - **Combination** of AT_BLAS + simdlen=256: not yet tested. AT_BLAS showed consistent latency improvement on ALL percentiles (p99 −3.6%). If you need to try one more InductorConfig option, try the combo: `{"max_autotune_gemm_backends": "AT_BLAS", "cpp.simdlen": 256}`. Low confidence (<3% expected) but the only untried combination.
   - `freezing`: allows torch.compile to treat model weights as constants, enabling more aggressive kernel fusion. This is a top-level torch.compile argument, not an inductor config key — check if vLLM exposes it.

5. **What did previous iterations reveal?**
   - Which bottleneck did each accepted fix remove?
   - Which hypothesis type was NOT yet tried this session?

### Step 2d — Generate Ranked Hypotheses

Produce 3–5 hypotheses. **All must be CodeEdit or InductorConfig.** Flags are exhausted — if you find yourself proposing a flag change, stop and re-examine the code scan.

```
Hypothesis N: <name>
  Type      : InductorConfig | CodeEdit
  Mechanism : WHY this increases tokens/sec — which specific op or bottleneck it addresses
  Evidence  : profiler data / code scan file:line / logical argument
  Fix       : file:line with specific change, OR exact inductor_compile_config JSON
  Est. gain : % estimate + confidence: low / medium / high
  Risk      : what could break; how to verify correctness after change
```

### Step 2e — Select

```
SELECTED: Hypothesis N — <name>
TYPE    : InductorConfig | CodeEdit
REASON  : <one sentence tying the hypothesis to profiler or code evidence>
```

**Default choice: the hypothesis most directly targeting the op that dominated the profiler output.** If two hypotheses target the same bottleneck, prefer the one that's lower-risk (smaller diff, easier to verify).

**Do not select Flag hypotheses.** If the profiler is unavailable and the code scan found nothing, the right action is to run Phase 4 — not to reach for a flag.

### Code Hypothesis Seed Catalog

Use when the code scan didn't surface an obvious target. Listed by the profiler op they address.

**If `aten::mm` / `aten::linear` dominates (GEMM-bound):**

| Priority | Type | Fix | Notes |
|---|---|---|---|
| 1 | CodeEdit | Enable weight-only INT8 quantization (AWQ CPU) | `cpu_wna16.py` implements CPU WNA16; check if Qwen2.5-0.5B has a compatible quantized checkpoint or can be quantized via `llm-compressor`. Halves weight memory → direct bandwidth reduction. |
| 2 | CodeEdit | Verify MergedColumnParallelLinear fuses QKV projection | `vllm/model_executor/layers/linear.py`: if QKV is fused into one matmul, check that the fused path is active on CPU and not falling back to 3 separate GEMMs. |
| 3 | InductorConfig | `{"max_autotune_gemm_backends": "AT_BLAS", "cpp.simdlen": 256}` (combo) | Individual options both tested and missed. Try the combo only if out of other ideas. Expected <3%, low confidence. |

**If `aten::scaled_dot_product_attention` dominates (attention-bound):**

| Priority | Type | Fix | Notes |
|---|---|---|---|
| 5 | CodeEdit | Check IPEX attention backend | `vllm/attention/backends/ipex_attn.py` — if Intel IPEX is installed, it provides optimized CPU attention. Check if it's selectable for ARM via `vllm/platforms/cpu.py`. |
| 6 | CodeEdit | Verify GQA is dispatched efficiently | Qwen2.5-0.5B has 2 KV heads vs 14 Q heads. Check the active attention backend's GQA path — some backends fall back to expanded KV (repeat_kv) which doubles KV memory reads. |
| 7 | InductorConfig | `"triton.cudagraphs": false` | Explicit disable to remove any graph-capture overhead on CPU (should already be disabled but verify). |

**If `aten::index_put_` / KV cache write dominates:**

| Priority | Type | Fix | Notes |
|---|---|---|---|
| 8 | CodeEdit | Investigate slot mapping kernel | `vllm/v1/worker/block_table.py` — `CPUModelRunner._postprocess_triton()` replaces the slot mapping with a CPU Triton kernel. Check `vllm/utils/cpu_triton_utils.py` for optimization opportunities. |
| 9 | CodeEdit | KV cache block size | Default is 16 tokens/block. A larger block reduces the number of scattered writes. Try `--block-size 32` only if profiler shows index_put dominates (previously tested at -0.4% when GEMM-bound — only revisit if regime changes). |

**Do Not Use (disproven):**

| Fix | Result | Why |
|---|---|---|
| `--dtype bfloat16` on ARM | **632× slower matmul** | M1/M2/M3/M4 AMX has no native bf16 compute |
| `--batch-size > 256` | Do not try | Latency ratio 1.91× at batch=256; linear scaling at larger batch |
| `OMP_NUM_THREADS=$(sysctl -n hw.physicalcpu)` | -2.3% | Hurts on M1 Pro; default threading better |
| `--block-size 32` | -0.4% (when GEMM-bound) | Only re-evaluate if profiler shifts to KV-write-bound |
| `inductor cpp.enable_concat_linear=true` | -4.6% + p90/p99 variance | Mid-run recompilation artifacts |
| Any Python scheduling change (`_prepare_inputs`, `np.repeat`, etc.) | -2.1% (noise) | Python overhead < 1% of step time at batch=256 |
| `copy_to_gpu()` identity guard | +0.5% (noise) | CPU `copy_(self)` is internally near-free |
| `--compilation-config '{"level":3}'` | Errors | No `level` field in CompilationConfig |
| `inductor max_autotune_gemm_backends=AT_BLAS` (alone) | +2.2% (below 3%) | Sub-threshold; LLVM already dispatches to Accelerate BLAS on ARM. All percentiles improved directionally (p99 −3.6%) but avg tok/s below acceptance bar. |
| `inductor cpp.simdlen=256` (alone) | +0.7% (noise) | LLVM auto-vectorization already handles NEON widths optimally; explicit hint adds no measurable gain. |

**Stop condition (any one):**
- `MAX_ITERS` (5) attempts completed
- 2 consecutive misses with no new profiler evidence
- All catalog hypotheses exhausted

When stalled: run Phase 4, then return to Phase 2 with the new profiler data.

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

### Step 3b — Smoke Test (mandatory for CodeEdit; recommended for Flag/InductorConfig)

Run a single iteration with no output-json to verify the model still produces output and does not crash:

```bash
VLLM_CPU_KVCACHE_SPACE=1 .venv/bin/vllm bench latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float32 \
  --batch-size 256 \
  --input-len 32 \
  --output-len 32 \
  --num-iters-warmup 0 \
  --num-iters 1
```

**Sanity-check the smoke-test latency against the baseline:**
- `tokens/sec ≈ batch × output_len / latency_seconds` — expect ~400-500 tok/s warmed
- Cold single-iter at batch=256 typically runs ~20s (no warmup). If >60s, something is wrong.
- **InductorConfig fixes change the compile cache key.** The first smoke-test iteration will recompile, inflating latency by 10–30%. This is expected — do not abort. Proceed to the full benchmark (with `--num-iters-warmup 2`) which will stabilize after warmup. Only abort if the smoke-test latency is >2× the baseline *and* is unlikely to be explained by compile overhead.

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
- batch=256: ~90s for warmup+5iters (2 warmup × ~18s + 5 bench × ~18s = ~126s total)

If a run takes >3× the expected time (~4.5 min), stop and investigate — likely wrong dtype or recompile triggered by an InductorConfig change (expected on first run; subsequent warmup iters stabilize).

```bash
VLLM_CPU_KVCACHE_SPACE=1 .venv/bin/vllm bench latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float32 \
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
# Commit everything (results + code edits)
git add perf_results/ docs/perf/
# For CodeEdit, also stage the changed source file(s):
# git add vllm/path/to/changed_file.py
git commit -m "perf(cpu): <one-line fix description>

type: <Flag|InductorConfig|CodeEdit>
tokens/sec: {baseline_tps} → {candidate_tps} (+{delta}%)
batch_size={N}, input_len=32, output_len=32"
# NEVER add Co-authored-by, Assisted-by, or any AI attribution trailers to commits.

COMMIT_SHA=$(git rev-parse --short HEAD)

# Promote candidate to new baseline
cp perf_results/candidate_N.json perf_results/baseline.json

# Append to log
# | N | <Flag/InductorConfig/Code> | <fix description> | {tok/s} | +{delta}% | {SHA or N/A} |
```

**Note:** `winning_config.sh` lives under `.cursor/` which is gitignored. Winning flags are reliably recorded only in `perf_results/run_log.jsonl` (which IS committed). Read the run log to reconstruct the winning invocation; do not rely on `winning_config.sh` across sessions.

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
  --batch-size 256 \
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
# NEVER add Co-authored-by, Assisted-by, or any AI attribution trailers to commits.
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

Winning flags are recorded in `perf_results/run_log.jsonl` (committed). `winning_config.sh` is gitignored and may be stale — use the run log as the source of truth.

---

## Key Source Files

| File | What to look for |
|---|---|
| `vllm/platforms/cpu.py` | Attention backend selection, dtype constraints, inductor config, compilation mode |
| `vllm/attention/backends/` | All attention backend implementations — find which one runs on CPU |
| `vllm/attention/layer.py` | Backend dispatch logic — how the active backend is chosen at runtime |
| `vllm/model_executor/layers/linear.py` | Linear layer forward; QKV fusion; which GEMM path is dispatched |
| `vllm/model_executor/layers/attention/` | Attention layer wrappers; GQA expand path |
| `vllm/model_executor/layers/quantization/cpu_wna16.py` | CPU WNA16 (AWQ) quantization implementation |
| `vllm/v1/worker/cpu_model_runner.py` | CPU-specific runner overrides; `_postprocess_triton()` for custom kernels |
| `vllm/utils/cpu_triton_utils.py` | CPU Triton kernel implementations (slot mapping, spec-decode helpers) |
| `tools/profiler/print_layerwise_table.py` | Profiler output analysis — **read this output before every hypothesis** |
| `.cursor/skills/vllm-perf/bench_compare.py` | This skill's benchmark harness |

**Do NOT spend time scanning these for new opportunities** (all confirmed below noise floor):
- `vllm/v1/worker/gpu_model_runner.py` — Python scheduling overhead < 1% of step time at batch=256
- `vllm/v1/core/sched/scheduler.py` — same
- `vllm/v1/utils.py` — `copy_to_gpu()` is a self-copy, internally free

---

## Empirical Results Log (accumulated across sessions)

Record tested hypotheses here so future sessions don't repeat them.

**Important:** this file is gitignored under `.cursor/`. Copy new rows into it at the end of every session. The machine-readable source of truth is `perf_results/run_log.jsonl` (committed).

**Sub-threshold signal convention:** if a fix improved ALL latency percentiles but tok/s stayed below 3%, mark it `~+N% (sub-threshold)`. Do not re-run the same fix alone; it is a candidate for combination only.

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
| 2026-05-09 | **batch=128 (from 64)** | **+24.3% ✓** | Still strongly sublinear (1.62× latency at 2× batch); much larger than "diminishing returns" prediction |
| 2026-05-09 | inductor cpp.enable_concat_linear=true | -4.6% ✗ | Causes mid-run recompilation bursts; p90 +13%, p99 +18%; avg tok/s regresses |
| 2026-05-09 | decode req_indices shortcut (skip np.repeat when tokens==1) | -2.1% ✗ (noise) | Python overhead < 1% of step time at batch=128; model is compute-bound, no Python fix is measurable |
| 2026-05-09 | **batch=256 (from 128)** | **+4.6% ✓** | Latency ratio 1.91× — above 1.7× stop threshold; batch scaling exhausted. Baseline now 463.6 tok/s. |
| 2026-05-09 | inductor max_autotune_gemm_backends=AT_BLAS | ~+2.2% (sub-threshold) | All percentiles improved (p50 −2.8%, p90 −3.0%, p99 −3.6%). Below 3% tok/s bar. Ran without profiler evidence — lesson: run Phase 4 first. |
| 2026-05-09 | inductor cpp.simdlen=256 | +0.7% ✗ (noise) | LLVM auto-vectorization already optimal on ARM NEON. Ran without profiler evidence. |
