# Pencarian Lokasi Strategis Gudang Logistik

Program ini menggunakan bahasa C dengan OpenCL untuk menjalankan perhitungan di GPU. Google Colab dipakai sebagai tempat compile dan run program.

## File Utama

```
src/main.c
```

Kernel OpenCL sudah ditanam langsung di dalam `main.c`, jadi tidak perlu file `.cl` terpisah.

## Rumus

```
Skor(G) = sum sqrt((Glat - Blat)^2 + (Glon - Blon)^2) * Jumlah_Penerima * Prioritas
```

Gudang dengan skor paling kecil adalah gudang paling strategis.

## Input

Program membutuhkan dua file CSV:

1. `bantuan_sosial.csv`

```csv
ID_Bantuan,Latitude,Longitude,Jumlah_Penerima,Prioritas
B00001,-7.390235,123.875466,3893,4
```

2. `kandidat_gudang.csv`

```csv
ID_Gudang,Latitude,Longitude,Kapasitas
G0001,-1.831314,102.332957,93697
```

## Cara Menjalankan di Google Colab

Aktifkan GPU dulu:

```
Runtime -> Change runtime type -> GPU
```

Upload file:

```python
from google.colab import files
files.upload()
```

Upload tiga file:

- `main.c`
- `bantuan_sosial.csv`
- `kandidat_gudang.csv`

Install header/library OpenCL:

```python
!apt-get update -qq
!apt-get install -y -qq ocl-icd-opencl-dev clinfo
```

Cek apakah GPU OpenCL terbaca:

```python
!clinfo | grep -E "Platform Name|Device Name|Device Type" | head -20
```

Compile:

```python
!gcc -O2 main.c -o gudang_opencl -lOpenCL -lm
```

Jalankan:

```python
!./gudang_opencl bantuan_sosial.csv kandidat_gudang.csv ranking_final.csv
```

Download hasil:

```python
from google.colab import files
files.download("ranking_final.csv")
```

## Output

Program menampilkan:

- platform OpenCL
- device GPU yang digunakan
- jumlah data yang dibaca
- waktu kernel GPU
- waktu total
- gudang dengan skor terbaik
- file ranking hasil akhir

File output:

```
ranking_final.csv
```

Kolom output:

```csv
Rank,ID_Gudang,Latitude,Longitude,Kapasitas,Skor_Strategis
```

## Struktur Folder

```
.
├── src/
│   └── main.c
├── docs/
│   ├── flowchart.md
│   ├── input_output.md
│   └── laporan.md
├── test/
│   └── benchmark_results.md
└── README.md
```

## Catatan Penting

Program ini sengaja memilih `CL_DEVICE_TYPE_GPU`. Jika Colab belum memakai runtime GPU atau OpenCL GPU tidak tersedia, program akan berhenti dengan pesan error.
