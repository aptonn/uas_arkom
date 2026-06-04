# Hasil Pengujian Program C/OpenCL GPU

## Lingkungan Pengujian

| Spesifikasi | Detail |
|---|---|
| Platform | Google Colaboratory |
| Runtime | GPU |
| GPU | Isi sesuai output program |
| Compiler | GCC |
| Library | OpenCL |
| File program | `main.c` |

## Perintah Pengujian

```python
!apt-get update -qq
!apt-get install -y -qq ocl-icd-opencl-dev clinfo
!gcc -O2 main.c -o gudang_opencl -lOpenCL -lm
!./gudang_opencl bantuan_sosial.csv kandidat_gudang.csv ranking_final.csv
```

## Hasil Eksekusi

Isi dari output program:

| Metrik | Nilai |
|---|---|
| Platform OpenCL | |
| Device GPU | |
| Jumlah titik bantuan | |
| Jumlah kandidat gudang | |
| Total operasi jarak | |
| Waktu kernel GPU | |
| Waktu total | |
| ID gudang terbaik | |
| Skor terbaik | |

## Top 10 Gudang Paling Strategis

Salin dari `ranking_final.csv`.

| Rank | ID_Gudang | Latitude | Longitude | Kapasitas | Skor |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | | | | | |

## Kesimpulan Pengujian

- Program berhasil membaca dataset CSV.
- Program berhasil menemukan GPU OpenCL.
- Program berhasil menjalankan kernel di GPU.
- File `ranking_final.csv` berhasil dibuat sebagai hasil akhir.
