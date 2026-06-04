# Dokumentasi Input dan Output

## Input 1: bantuan_sosial.csv

Kolom:

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `ID_Bantuan` | string | ID titik bantuan |
| `Latitude` | float | Koordinat lintang |
| `Longitude` | float | Koordinat bujur |
| `Jumlah_Penerima` | int | Jumlah penerima bantuan |
| `Prioritas` | int | Tingkat prioritas |

Contoh:

```csv
ID_Bantuan,Latitude,Longitude,Jumlah_Penerima,Prioritas
B00001,-7.390235,123.875466,3893,4
```

## Input 2: kandidat_gudang.csv

Kolom:

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `ID_Gudang` | string | ID kandidat gudang |
| `Latitude` | float | Koordinat lintang |
| `Longitude` | float | Koordinat bujur |
| `Kapasitas` | int | Kapasitas gudang |

Contoh:

```csv
ID_Gudang,Latitude,Longitude,Kapasitas
G0001,-1.831314,102.332957,93697
```

## Output: ranking_final.csv

Kolom:

| Kolom | Deskripsi |
|---|---|
| `Rank` | Urutan ranking |
| `ID_Gudang` | ID kandidat gudang |
| `Latitude` | Koordinat lintang gudang |
| `Longitude` | Koordinat bujur gudang |
| `Kapasitas` | Kapasitas gudang |
| `Skor_Strategis` | Total jarak Euclidean berbobot |

Contoh:

```csv
Rank,ID_Gudang,Latitude,Longitude,Kapasitas,Skor_Strategis
1,G0007,-3.123456,117.654321,85000,12345678.0000
2,G0012,-2.987654,118.123456,67000,12456789.0000
```

Nilai `Skor_Strategis` yang lebih kecil berarti lokasi gudang lebih strategis.

## Perintah Colab

Compile:

```python
!apt-get update -qq
!apt-get install -y -qq ocl-icd-opencl-dev clinfo
!gcc -O2 main.c -o gudang_opencl -lOpenCL -lm
```

Jalankan:

```python
!./gudang_opencl bantuan_sosial.csv kandidat_gudang.csv ranking_final.csv
```
