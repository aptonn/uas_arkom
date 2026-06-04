# Dokumentasi Input & Output

## Input

### bantuan_sosial.csv
| Kolom | Tipe | Contoh |
|---|---|---|
| ID_Bantuan | string | B00001 |
| Latitude | float | -7.3902 |
| Longitude | float | 123.875 |
| Jumlah_Penerima | int | 3893 |
| Prioritas | int (1–5) | 4 |

**Ukuran**: 50.000 baris

### kandidat_gudang.csv
| Kolom | Tipe | Contoh |
|---|---|---|
| ID_Gudang | string | G0001 |
| Latitude | float | -1.8313 |
| Longitude | float | 102.332 |
| Kapasitas | int | 93697 |

**Ukuran**: 1.000 baris

---

## Output (dicetak ke terminal)

```
=============================================================
  Brute-Force Euclidean — Pencarian Gudang Strategis
  Sequential | OpenMP | OpenCL
=============================================================

[1] Memuat data...
    Penerima bantuan : 50000 titik
    Kandidat gudang  : 1000 lokasi
    Total operasi    : 50000000 pasang jarak

[2] Sequential...
    Waktu          : X.XXXX detik
    Gudang terbaik : G0360 (skor = 4868027392.00)

[3] OpenMP (N thread tersedia)...
    Waktu          : X.XXXX detik
    Speedup        : X.XXx dari sequential
    Gudang terbaik : G0360 (skor = 4868027392.00)

[4] OpenCL...
  Device : NVIDIA Tesla T4 (40 compute units)
  Global size: 1024 | Workgroup: 64 | Workgroups: 16
    Waktu          : X.XXXX detik
    Speedup        : X.XXx dari sequential
    Gudang terbaik : G0360 (skor = ...)

=============================================================
  BENCHMARK SUMMARY
=============================================================
  Metode                    Waktu (s)      Speedup
  Sequential                   XX.XXXX 1.00x (baseline)
  OpenMP (CPU)                  X.XXXX       X.XXx
  OpenCL (GPU)                  X.XXXX       X.XXx

  TOP 5 GUDANG PALING STRATEGIS:

  Rank 1   | G0360    | lat= -1.8241 | lon= 117.6470 | ...
  ...
```
