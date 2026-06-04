# Flowchart & Diagram Sistem

## Alur Utama Program

```
MULAI
  │
  ▼
Baca bantuan_sosial.csv     ← 50.000 baris
Baca kandidat_gudang.csv    ← 1.000 baris
  │
  ▼
Hitung bobot[i] = Penerima[i] × Prioritas[i]
  │
  ├──────────────────────────────────────┐
  │                                      │
  ▼                                      ▼
[Sequential]                         [OpenMP]
for i = 0..999:                      #pragma omp parallel for
  for j = 0..49999:                  for i = 0..999: (dibagi ke N thread)
    dist = √(Δlat²+Δlon²)             for j = 0..49999:
    total += dist × bobot[j]            dist = √(Δlat²+Δlon²)
  skor_seq[i] = total                   total += dist × bobot[j]
                                       skor_omp[i] = total
  │                                      │
  └───────────────┬──────────────────────┘
                  │
                  ▼
             [OpenCL]
             Transfer data ke GPU
             Compile kernel
             Launch 1.000 work-items (thread GPU)
               ┌─────────────────────────────────┐
               │  __kernel hitung_skor:          │
               │  gid = get_global_id(0)         │
               │  for j = 0..49999:              │
               │    dist = native_sqrt(...)      │
               │    total += dist × bobot[j]     │
               │  skor_ocl[gid] = total          │
               └─────────────────────────────────┘
             Baca hasil dari GPU
                  │
                  ▼
             Cari min(skor) → gudang terbaik
             Cetak benchmark & top-5
                  │
                  ▼
               SELESAI
```

## Model Eksekusi OpenCL

```
CPU (Host)                          GPU (Device)
──────────────────────────────────────────────────

Data di RAM:                        Global Memory GPU:
  g_lat[1000]          ──copy──►    g_lat[1000]
  g_lon[1000]          ──copy──►    g_lon[1000]
  b_lat[50000]         ──copy──►    b_lat[50000]
  b_lon[50000]         ──copy──►    b_lon[50000]
  b_bobot[50000]       ──copy──►    b_bobot[50000]
                                    skor[1000]  ← ditulis kernel

Launch kernel: global_size=1024, local_size=64

                                    Workgroup 0 (thread 0–63)
                                      → proses gudang 0–63
                                    Workgroup 1 (thread 0–63)
                                      → proses gudang 64–127
                                    ...
                                    Workgroup 15 (thread 0–63)
                                      → proses gudang 960–999
                                      (thread 1000–1023: return langsung)

Baca hasil:         skor[1000]  ◄──  skor[1000]
Cari min(skor) di CPU
```
