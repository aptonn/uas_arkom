# Hasil Pengujian & Benchmark

## Lingkungan Pengujian
| Spesifikasi | Detail |
|---|---|
| Platform | Google Colaboratory |
| Runtime | GPU (NVIDIA T4 / A100 / V100) |
| OS | Ubuntu |
| Compiler | GCC 11+ dengan `-O2 -fopenmp` |
| OpenCL | 1.2 (NVIDIA CUDA) |

---

## Hasil Benchmark

| Metode | Waktu (s) | Speedup |
|---|---|---|
| Sequential | | 1.00× |
| OpenMP (N thread) | | |
| OpenCL (GPU) | | |

---

## Gudang Paling Strategis

| Rank | ID | Lat | Lon | Kapasitas | Skor |
|---|---|---|---|---|---|
| 1 | G0360 | -1.8241 | 117.6470 | 88516 | 4868027392 |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

---

## Analisis
- Speedup OpenCL tidak linier karena ada overhead transfer data PCIe (CPU→GPU).
- Kernel murni komputasi jauh lebih cepat dari wall time (termasuk transfer).
- OpenMP speedup ≈ jumlah core yang tersedia (2 vCPU di Colab).
