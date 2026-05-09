# vLLM CPU Optimization Log

**Model:** Qwen/Qwen2.5-0.5B-Instruct | **Device:** cpu | **in=32 out=32**

| # | Type | Fix | tok/s | Delta | Commit |
|---|------|-----|-------|-------|--------|
| 0 | flag | Baseline (bfloat16, batch=4, in=32, out=32) | 42.0 | — | N/A |
| 1 | flag | batch=8 | 60.3 | +43.6% | N/A (flag-only) |
| 2 | flag | OMP_NUM_THREADS=8 | — | -2.3% (miss) | reverted |
| 3 | flag | block-size=32 | — | -0.4% (miss) | reverted |
| 4 | flag | batch=16 | 68.0 | +12.8% | N/A (flag-only) |
| 5 | code | Skip no-op copy_to_gpu() identity guard | — | +0.5% (miss) | reverted |
| 6 | flag | batch=32 | 72.7 | +7.8% | dab1164 |
| 7 | flag | batch=64 | 76.6 | +5.4% | a1d9863 |
| 8 | flag | dtype=float32 (from bfloat16) | 356.7 | +365.6% | 6218c24 |
| 9 | flag | batch=128 | 443.3 | +24.3% | d7f0e66 |
| 10 | inductor | cpp.enable_concat_linear=true | — | -4.6% (miss) | cd9a938 |
| 11 | code | decode req_indices shortcut (skip np.repeat) | — | -2.1% (noise; compute-bound) | abd11fb |
| 12 | flag | batch=256 | 463.6 | +4.6% | aac63aa (latency ratio 1.91× — plateau) |
| 13 | inductor | max_autotune_gemm_backends=AT_BLAS | — | ~+2.2% (below 3% threshold) | reverted |
| 14 | inductor | cpp.simdlen=256 | — | +0.7% (noise) | reverted |
| 15 | code | apply_temperature skip when T=1.0 | 480.1 | +3.6% | 6cd1a0e |
