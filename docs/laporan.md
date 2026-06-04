# Laporan Singkat — UAS Arsitektur dan Sistem Komputer

## Judul
Akselerasi Pencarian Lokasi Strategis Gudang Logistik Bantuan Sosial  
Menggunakan Algoritma Brute-Force Jarak Euclidean Paralel Berbasis OpenCL

---

## 1. Deskripsi Masalah

Terdapat **50.000 titik penerima bantuan sosial** dan **1.000 kandidat gudang logistik**.  
Tugas: temukan gudang yang paling strategis — paling efisien menjangkau semua penerima.

**Formula skor strategis per gudang G:**

```
Skor(G) = Σᵢ [ √((Glat − Blat_i)² + (Glon − Blon_i)²) × Penerima_i × Prioritas_i ]
```

- Gudang dengan **skor terkecil** = paling strategis.
- Total komputasi: 1.000 × 50.000 = **50 juta operasi jarak** → ideal untuk GPU.

---

## 2. Tiga Metode Komputasi

| Metode | Hardware | Cara Kerja |
|---|---|---|
| Sequential | 1 CPU core | Loop gudang satu per satu |
| OpenMP | Multi-core CPU | `#pragma omp parallel for` — loop gudang dibagi ke semua core |
| OpenCL | GPU (ribuan core) | 1 work-item = 1 gudang, semua jalan bersamaan |

---

## 3. Desain Keputusan OpenCL

- **Granularitas**: 1 work-item per gudang. 1.000 work-item berjalan serentak.
- **Workgroup size = 64**: kelipatan 32 (warp NVIDIA), umum digunakan untuk NVIDIA GPU di Colab.
- **Guard di kernel**: `if (gid >= n_gudang) return;` — karena global_size dibulatkan ke atas.
- **`native_sqrt`**: lebih cepat dari `sqrt` standar di GPU, akurasi float32 sudah cukup untuk koordinat.
- **Float32**: throughput GPU untuk float32 jauh lebih tinggi dari float64.

---

## 4. Mengapa Speedup Tidak Linier?

Sesuai **Hukum Amdahl**: bagian serial membatasi speedup maksimum.

Bagian yang tidak bisa diparalelkan:
1. Baca CSV → sekuensial
2. Transfer data CPU → GPU (PCIe bottleneck)
3. Cari minimum skor → sekuensial kecil

Faktor lain:
- **Memory bottleneck**: 50.000 titik bantuan harus dibaca ulang oleh setiap work-item (tidak di-cache secara lokal).
- **Kernel launch overhead**: ada latency saat GPU "menyalakan" ribuan thread.

---

## 5. Dataset

| File | Baris | Kolom |
|---|---|---|
| `bantuan_sosial.csv` | 50.000 | ID, Latitude, Longitude, Jumlah_Penerima, Prioritas |
| `kandidat_gudang.csv` | 1.000 | ID, Latitude, Longitude, Kapasitas |

Sebaran koordinat: Indonesia (Lat ≈ −11° s/d +6°, Lon ≈ +95° s/d +141°).

---

## 6. Hasil Pengujian

*(isi dari output Google Colab setelah program dijalankan)*

| Metode | Waktu (s) | Speedup |
|---|---|---|
| Sequential | | 1.00x |
| OpenMP | | |
| OpenCL | | |

**Gudang paling strategis**: G0360 (lat ≈ −1.82, lon ≈ 117.65)
