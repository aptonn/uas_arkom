# Akselerasi Pencarian Lokasi Strategis Gudang Logistik Bantuan Sosial
## Brute-Force Euclidean Distance: Sequential | OpenMP | OpenCL

---

## Nama Penyusun
*Afton Ilman Firmansyah, Aunal Ghoribi Prasetyo, Sultan Hakim Sukmana*

---

## Deskripsi Proyek

Proyek ini mencari lokasi gudang logistik **paling strategis** dari 1.000 kandidat lokasi, dievaluasi terhadap 50.000 titik penerima bantuan sosial menggunakan algoritma brute-force jarak Euclidean dengan akselerasi GPU via OpenCL.

**Formula skor:**
```
Skor(G) = Σ [ √((Glat−Blat)² + (Glon−Blon)²) × Jumlah_Penerima × Prioritas ]
```
Gudang dengan skor terkecil = paling strategis.

**Fitur:**
- Implementasi **Sequential** (baseline CPU single-thread)
- Implementasi **OpenMP** (CPU multi-thread, `#pragma omp parallel for`)
- Implementasi **OpenCL** (GPU paralel, 1 work-item per gudang)
- Benchmark otomatis: waktu & speedup ketiga metode
- Validasi konsistensi hasil antara ketiga metode
- Tampilkan top-5 gudang paling strategis

---

## Cara Menjalankan di Google Colab

**1. Ubah runtime ke GPU:**
Runtime → Change runtime type → **GPU**

**2. Upload file ke Colab:**
```python
Upload secara manual lewat sidebar
# Upload: main.c, bantuan_sosial.csv, kandidat_gudang.csv
```

**3. Install OpenCL headers & compile:**
```bash
!apt-get install -y opencl-headers ocl-icd-opencl-dev -q
!gcc -O2 -fopenmp main.c -o gudang -lOpenCL -lm
```

**4. Jalankan:**
```bash
!./gudang bantuan_sosial.csv kandidat_gudang.csv
```

> Jika tidak ada GPU (CPU runtime), tambahkan dulu:
> `!apt-get install -y pocl-opencl-icd -q`

---

## Struktur Repository
```
src/main.c                  ← satu file C (semua implementasi)
docs/laporan.md             ← laporan singkat
docs/flowchart.md           ← diagram sistem
docs/input_output.md        ← dokumentasi I/O
test/benchmark_results.md   ← hasil pengujian
data/bantuan_sosial.csv     ← dataset penerima bantuan (50.000 baris)
data/kandidat_gudang.csv    ← dataset kandidat gudang (1.000 baris)
```

---

## Link Video
https://youtu.be/uQKTD6AMo-s?feature=shared

