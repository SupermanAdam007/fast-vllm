# vLLM CPU Optimization Log

**Model:** Qwen/Qwen2.5-0.5B-Instruct | **Device:** cpu | **batch=8** | **in=128 out=128**

| # | Fix | tok/s | Delta | Commit |
|---|-----|-------|-------|--------|
| 0 | Baseline (bfloat16, batch=4, in=32, out=32) | 42.0 | — | N/A |
| 1 | batch=8 (bench_compare bug: latency false-positive) | 60.3 | +43.6% | N/A (flag-only) |
| 2 | OMP_NUM_THREADS=8 | — | -2.3% (miss) | reverted |
| 3 | block-size=32 | — | -0.4% (miss) | reverted |
| 4 | batch=16 | 68.0 | +12.8% | N/A (flag-only) |
