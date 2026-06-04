# Flowchart Sistem

## Alur Program C/OpenCL

```
MULAI
  |
  v
Baca bantuan_sosial.csv di CPU
  |
  v
Baca kandidat_gudang.csv di CPU
  |
  v
Hitung bobot tiap titik bantuan
bobot = Jumlah_Penerima * Prioritas
  |
  v
Pilih device GPU OpenCL
  |
  v
Buat buffer GPU
  |
  v
Kirim data ke GPU
  |
  v
Compile kernel OpenCL
  |
  v
Jalankan kernel:
1 work-item = 1 kandidat gudang
  |
  v
Ambil skor dari GPU ke CPU
  |
  v
Cari skor paling kecil
  |
  v
Tampilkan gudang terbaik
  |
  v
Simpan ranking_final.csv
  |
  v
SELESAI
```

## Model Eksekusi OpenCL

```
GPU work-items

work-item 0    -> Gudang 0    -> loop semua titik bantuan
work-item 1    -> Gudang 1    -> loop semua titik bantuan
work-item 2    -> Gudang 2    -> loop semua titik bantuan
...
work-item 999  -> Gudang 999  -> loop semua titik bantuan
```

## Kernel OpenCL

Bagian inti kernel:

```c
__kernel void hitung_skor_gudang(
    __global const float *b_lat,
    __global const float *b_lon,
    __global const float *b_weight,
    __global const float *g_lat,
    __global const float *g_lon,
    __global float *scores,
    const int n_bantuan,
    const int n_gudang)
{
    int gid = get_global_id(0);
    if (gid >= n_gudang) return;

    float lat_g = g_lat[gid];
    float lon_g = g_lon[gid];
    float total = 0.0f;

    for (int i = 0; i < n_bantuan; i++) {
        float dlat = lat_g - b_lat[i];
        float dlon = lon_g - b_lon[i];
        float dist = sqrt(dlat * dlat + dlon * dlon);
        total += dist * b_weight[i];
    }

    scores[gid] = total;
}
```
