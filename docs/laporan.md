# Laporan Singkat Proyek UAS Arsitektur dan Sistem Komputer

## Judul

Pencarian Lokasi Strategis Gudang Logistik Bantuan Sosial Menggunakan C dan OpenCL GPU

## 1. Deskripsi Masalah

Program memilih 1 gudang terbaik dari beberapa kandidat gudang. Setiap kandidat dihitung skornya terhadap semua titik penerima bantuan sosial.

Gudang dianggap paling strategis jika memiliki total jarak berbobot paling kecil.

Rumus:

```
Skor(G) = sum sqrt((Glat - Blat)^2 + (Glon - Blon)^2) * Jumlah_Penerima * Prioritas
```

Keterangan:

- `Glat`, `Glon`: koordinat kandidat gudang
- `Blat`, `Blon`: koordinat titik bantuan
- `Jumlah_Penerima`: jumlah penerima bantuan di titik tersebut
- `Prioritas`: tingkat prioritas bantuan

## 2. Implementasi

Bahasa yang digunakan:

```
C + OpenCL
```

File program:

```
src/main.c
```

Program terdiri dari dua bagian:

- Host program C: membaca CSV, menyiapkan buffer OpenCL, menjalankan kernel, menyimpan ranking.
- Kernel OpenCL: menghitung skor setiap kandidat gudang di GPU.

Kernel OpenCL ditanam langsung sebagai string di dalam `main.c`, sehingga pengguna hanya perlu compile satu file C.

## 3. Strategi Paralel GPU

Satu work-item OpenCL menangani satu kandidat gudang.

```
work-item 0 -> hitung skor gudang 0
work-item 1 -> hitung skor gudang 1
work-item 2 -> hitung skor gudang 2
...
```

Setiap work-item melakukan loop ke seluruh titik bantuan:

```c
for (int i = 0; i < n_bantuan; i++) {
    float dlat = lat_g - b_lat[i];
    float dlon = lon_g - b_lon[i];
    float dist = sqrt(dlat * dlat + dlon * dlon);
    total += dist * b_weight[i];
}
```

Karena skor antar gudang tidak saling bergantung, proses ini cocok dijalankan secara paralel di GPU.

## 4. Input

Program membaca dua file CSV:

- `bantuan_sosial.csv`
- `kandidat_gudang.csv`

Format detail ada di [input_output.md](input_output.md).

## 5. Output

Program menghasilkan file:

```
ranking_final.csv
```

Isi file adalah ranking gudang dari skor terkecil sampai terbesar. Ranking pertama adalah gudang paling strategis.

## 6. Cara Menjalankan di Google Colab

Aktifkan GPU:

```
Runtime -> Change runtime type -> GPU
```

Install OpenCL:

```python
!apt-get update -qq
!apt-get install -y -qq ocl-icd-opencl-dev clinfo
```

Compile dan jalankan:

```python
!gcc -O2 main.c -o gudang_opencl -lOpenCL -lm
!./gudang_opencl bantuan_sosial.csv kandidat_gudang.csv ranking_final.csv
```

## 7. Kelebihan

- Perhitungan utama berjalan di GPU.
- Tidak menggunakan OpenMP/CPU parallel.
- Tetap memakai bahasa C untuk host program.
- Tidak membutuhkan Python sebagai implementasi algoritma.
- File kode utama hanya satu: `main.c`.

## 8. Keterbatasan

- Tetap ada proses CPU untuk membaca CSV, membuat buffer, dan menyimpan output.
- OpenCL runtime harus tersedia di Colab.
- Jarak Euclidean pada koordinat latitude/longitude bukan jarak geografis paling akurat.
- Kapasitas gudang hanya ditampilkan, belum dipakai sebagai batas pemilihan.
