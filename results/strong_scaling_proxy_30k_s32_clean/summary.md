# 30k + S32 sample-scaling run

Device: NVIDIA GeForce RTX 5070 Ti (16303 MiB). Configuration: 36470 cells,
1088 directions, 14 SI iterations, RSI tailExtra=10, and
RSI_CUDA_RSI_MAX_SAMPLES_PER_BATCH=1024. Each case ran alone on the GPU.

| Samples | CUDA total (s) | RSI total (s) | RSI sweep (s) | Wall (s) | GPU-Util avg (%) | GPU-Util range (%) | Peak memory (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 98.393 | 56.222 | 55.020 | 99.02 | 99.73 | 98-100 | 4582 |
| 2000 | 265.899 | 187.251 | 183.246 | 263.55 | 99.77 | 98-100 | 7566 |
| 4000 | 471.643 | 389.662 | 380.564 | 465.97 | 99.24 | 98-100 | 7607 |
| 8000 | 610.210 | 526.444 | 518.168 | 606.16 | 99.36 | 98-100 | 7631 |
| 16000 | 1237.560 | 1190.700 | 1166.410 | 1217.23 | 99.54 | 53-100 | 12432 |

`GPU-Util` is the one-second device-busy sample reported by `nvidia-smi`; it
is not Nsight Compute achieved occupancy. The 16000-sample minimum includes
the end-of-run/output interval. Per-case raw timing is in `s*/run.log` and
the one-second utilization samples are in `s*/gpu.csv`.
