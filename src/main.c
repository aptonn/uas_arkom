/* =============================================================
 * main.c — Akselerasi Pencarian Lokasi Strategis Gudang Logistik
 *           Brute-Force Jarak Euclidean: Sequential | OpenMP | OpenCL
 *
 * Cara compile di Google Colab:
 *   !apt-get install -y opencl-headers ocl-icd-opencl-dev -q
 *   !gcc -O2 -fopenmp main.c -o gudang -lOpenCL -lm
 *
 * Cara jalankan:
 *   !./gudang bantuan_sosial.csv kandidat_gudang.csv
 *
 * (Jika tidak ada GPU, install dulu: !apt-get install -y pocl-opencl-icd -q)
 * ============================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Target OpenCL 1.2 — paling kompatibel dengan GPU Colab & pocl */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

/* ─── KONFIGURASI ─── */
#define MAX_BANTUAN  51000   /* lebih dari 50000 biar aman */
#define MAX_GUDANG    1100
#define MAX_LINE       300
#define WG_SIZE         64   /* workgroup size; kelipatan 32 untuk NVIDIA */
#define TOP_N            5   /* jumlah gudang terbaik yang ditampilkan */

/* =============================================================
 * KERNEL OPENCL — disimpan sebagai string di dalam program
 *
 * Cara kerjanya:
 *   Setiap "work-item" (thread GPU) menangani SATU gudang.
 *   Thread itu lalu ngitung jarak ke SEMUA 50.000 titik bantuan.
 *   Semua 1.000 thread jalan BERSAMAAN di GPU.
 *
 *   get_global_id(0) = "saya thread ke berapa?" = indeks gudang saya
 * ============================================================= */
static const char *kernel_src =
"__kernel void hitung_skor(                                         \n"
"    __global const float* g_lat,   /* lat semua gudang */         \n"
"    __global const float* g_lon,   /* lon semua gudang */         \n"
"    __global const float* b_lat,   /* lat semua bantuan */        \n"
"    __global const float* b_lon,   /* lon semua bantuan */        \n"
"    __global const float* b_bobot, /* penerima * prioritas */     \n"
"    __global       float* skor,    /* output: skor tiap gudang */ \n"
"    const int n_bantuan,                                          \n"
"    const int n_gudang)                                           \n"
"{                                                                  \n"
"    int gid = get_global_id(0);                                   \n"
"    if (gid >= n_gudang) return; /* thread ekstra langsung keluar */\n"
"                                                                   \n"
"    float lat_g = g_lat[gid];                                     \n"
"    float lon_g = g_lon[gid];                                     \n"
"    float total = 0.0f;                                           \n"
"                                                                   \n"
"    /* Brute-force: hitung jarak ke semua titik bantuan */        \n"
"    for (int i = 0; i < n_bantuan; i++) {                        \n"
"        float dlat = lat_g - b_lat[i];                           \n"
"        float dlon = lon_g - b_lon[i];                           \n"
"        float dist = native_sqrt(dlat*dlat + dlon*dlon);         \n"
"        total += dist * b_bobot[i];                               \n"
"    }                                                              \n"
"    skor[gid] = total;                                            \n"
"}                                                                  \n";

/* =============================================================
 * ARRAY DATA (global, dipakai ketiga metode)
 * ============================================================= */
static float b_lat[MAX_BANTUAN];    /* latitude penerima bantuan  */
static float b_lon[MAX_BANTUAN];    /* longitude penerima bantuan */
static float b_bobot[MAX_BANTUAN];  /* bobot = jumlah_penerima * prioritas */

static float g_lat[MAX_GUDANG];     /* latitude kandidat gudang  */
static float g_lon[MAX_GUDANG];     /* longitude kandidat gudang */
static int   g_kap[MAX_GUDANG];     /* kapasitas gudang          */
static char  g_id[MAX_GUDANG][16];  /* ID gudang (e.g. "G0001")  */

/* Skor hasil tiap metode (disimpan terpisah untuk verifikasi) */
static float skor_seq[MAX_GUDANG];
static float skor_omp[MAX_GUDANG];
static float skor_ocl[MAX_GUDANG];

static int n_bantuan = 0;
static int n_gudang  = 0;

/* =============================================================
 * BACA CSV
 * ============================================================= */
int baca_bantuan(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: tidak bisa buka '%s'\n", path);
        return -1;
    }

    char line[MAX_LINE], id[24];
    float lat, lon;
    int penerima, prioritas;

    if (!fgets(line, MAX_LINE, fp)) { fclose(fp); return 0; } /* skip header */

    while (fgets(line, MAX_LINE, fp) && n_bantuan < MAX_BANTUAN) {
        if (sscanf(line, "%23[^,],%f,%f,%d,%d",
                   id, &lat, &lon, &penerima, &prioritas) == 5) {
            b_lat[n_bantuan]   = lat;
            b_lon[n_bantuan]   = lon;
            b_bobot[n_bantuan] = (float)penerima * (float)prioritas;
            n_bantuan++;
        }
    }
    fclose(fp);
    return n_bantuan;
}

int baca_gudang(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: tidak bisa buka '%s'\n", path);
        return -1;
    }

    char line[MAX_LINE], id[16];
    float lat, lon;
    int kap;

    if (!fgets(line, MAX_LINE, fp)) { fclose(fp); return 0; } /* skip header */

    while (fgets(line, MAX_LINE, fp) && n_gudang < MAX_GUDANG) {
        if (sscanf(line, "%15[^,],%f,%f,%d",
                   id, &lat, &lon, &kap) == 4) {
            strncpy(g_id[n_gudang], id, 15);
            g_lat[n_gudang] = lat;
            g_lon[n_gudang] = lon;
            g_kap[n_gudang] = kap;
            n_gudang++;
        }
    }
    fclose(fp);
    return n_gudang;
}

/* =============================================================
 * [METODE 1] SEQUENTIAL
 *
 * Analogi: satu orang ngerjain semua 1.000 gudang satu per satu.
 * Setiap gudang: ngitung jarak ke 50.000 titik bantuan.
 * ============================================================= */
double hitung_sequential(void) {
    double t_mulai = omp_get_wtime();

    for (int i = 0; i < n_gudang; i++) {
        float total  = 0.0f;
        float lat_g  = g_lat[i];
        float lon_g  = g_lon[i];

        for (int j = 0; j < n_bantuan; j++) {
            float dlat = lat_g - b_lat[j];
            float dlon = lon_g - b_lon[j];
            float dist = sqrtf(dlat*dlat + dlon*dlon);
            total += dist * b_bobot[j];
        }
        skor_seq[i] = total;
    }

    return omp_get_wtime() - t_mulai;
}

/* =============================================================
 * [METODE 2] OPENMP — CPU Paralel
 *
 * Analogi: N orang (= N core CPU) membagi 1.000 gudang rata.
 * Kalau ada 8 core, tiap orang ngerjain 125 gudang sekaligus.
 *
 * #pragma omp parallel for = "bagikan loop ini ke semua thread"
 * schedule(dynamic, 8)    = thread ambil 8 gudang sekaligus,
 *                           kalau sudah selesai ambil lagi.
 * ============================================================= */
double hitung_openmp(void) {
    double t_mulai = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < n_gudang; i++) {
        float total = 0.0f;
        float lat_g = g_lat[i];
        float lon_g = g_lon[i];

        for (int j = 0; j < n_bantuan; j++) {
            float dlat = lat_g - b_lat[j];
            float dlon = lon_g - b_lon[j];
            float dist = sqrtf(dlat*dlat + dlon*dlon);
            total += dist * b_bobot[j];
        }
        skor_omp[i] = total;
    }

    return omp_get_wtime() - t_mulai;
}

/* =============================================================
 * [METODE 3] OPENCL — GPU Paralel
 *
 * Analogi: ribuan robot mini (GPU core) jalan bersamaan.
 * Setiap robot = 1 work-item = tangani 1 gudang.
 * 1.000 gudang dikerjakan SERENTAK di GPU.
 *
 * Alur:
 *   1. Kirim data dari RAM ke GPU memory (clCreateBuffer)
 *   2. Compile kernel (clBuildProgram)
 *   3. Jalankan kernel (clEnqueueNDRangeKernel)
 *   4. Baca hasil balik dari GPU (clEnqueueReadBuffer)
 * ============================================================= */
double hitung_opencl(void) {
    cl_int err;

    /* --- Langkah 1: Cari platform & device --- */
    cl_platform_id platform;
    cl_uint n_platform;
    err = clGetPlatformIDs(1, &platform, &n_platform);
    if (err != CL_SUCCESS || n_platform == 0) {
        printf("  [!] Tidak ada OpenCL platform ditemukan.\n");
        return -1.0;
    }

    /* Coba GPU dulu, kalau tidak ada pakai CPU */
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            printf("  [!] Tidak ada OpenCL device (GPU/CPU).\n");
            return -1.0;
        }
        printf("  [i] Fallback ke OpenCL CPU (install pocl jika ingin GPU).\n");
    }

    char dev_name[128] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    cl_uint cu = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    printf("  Device : %s (%u compute units)\n", dev_name, cu);

    /* --- Langkah 2: Buat context dan command queue --- */
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    /* clCreateCommandQueue sudah deprecated di OpenCL 2.0,
       tapi masih berfungsi di semua versi — paling kompatibel */
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    #pragma GCC diagnostic pop

    /* --- Langkah 3: Compile kernel dari string --- */
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    err = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        /* Tampilkan pesan error kompilasi kernel */
        char build_log[4096];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(build_log), build_log, NULL);
        fprintf(stderr, "  [!] Kernel build error:\n%s\n", build_log);
        clReleaseProgram(prog);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return -1.0;
    }
    cl_kernel kernel = clCreateKernel(prog, "hitung_skor", &err);

    /* --- Langkah 4: Kirim data ke GPU memory --- */
    size_t sz_b = (size_t)n_bantuan * sizeof(float); /* ukuran array bantuan */
    size_t sz_g = (size_t)n_gudang  * sizeof(float); /* ukuran array gudang  */

    /* CL_MEM_COPY_HOST_PTR = langsung copy dari RAM ke GPU saat buffer dibuat */
    cl_mem buf_g_lat   = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz_g, g_lat,   &err);
    cl_mem buf_g_lon   = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz_g, g_lon,   &err);
    cl_mem buf_b_lat   = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz_b, b_lat,   &err);
    cl_mem buf_b_lon   = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz_b, b_lon,   &err);
    cl_mem buf_b_bobot = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz_b, b_bobot, &err);
    cl_mem buf_skor    = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz_g, NULL, &err);

    /* --- Langkah 5: Set argumen kernel --- */
    int nb = n_bantuan;
    int ng = n_gudang;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_g_lat);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_g_lon);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_b_lat);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_b_lon);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_b_bobot);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &buf_skor);
    clSetKernelArg(kernel, 6, sizeof(int),    &nb);
    clSetKernelArg(kernel, 7, sizeof(int),    &ng);

    /*
     * global_size = jumlah total work-item yang diluncurkan
     * Harus kelipatan WG_SIZE (64), dan >= n_gudang
     * Contoh: n_gudang=1000, WG_SIZE=64 -> global_size=1024
     *         -> 16 workgroup, masing-masing 64 thread
     *         -> thread 1000-1023 langsung return (ada guard di kernel)
     */
    size_t gs = ((size_t)(n_gudang + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
    size_t ls = WG_SIZE;
    printf("  Global size: %zu | Workgroup: %d | Workgroups: %zu\n",
           gs, WG_SIZE, gs / WG_SIZE);

    /* --- Langkah 6: Jalankan kernel dan ukur waktu --- */
    double t_mulai = omp_get_wtime();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gs, &ls, 0, NULL, NULL);
    clFinish(queue); /* tunggu sampai GPU selesai */
    double elapsed = omp_get_wtime() - t_mulai;

    if (err != CL_SUCCESS) {
        fprintf(stderr, "  [!] Error saat enqueue kernel: %d\n", err);
        elapsed = -1.0;
    } else {
        /* --- Langkah 7: Baca hasil dari GPU ke RAM --- */
        clEnqueueReadBuffer(queue, buf_skor, CL_TRUE, 0, sz_g, skor_ocl, 0, NULL, NULL);
    }

    /* --- Bersihkan resource GPU --- */
    clReleaseMemObject(buf_g_lat);
    clReleaseMemObject(buf_g_lon);
    clReleaseMemObject(buf_b_lat);
    clReleaseMemObject(buf_b_lon);
    clReleaseMemObject(buf_b_bobot);
    clReleaseMemObject(buf_skor);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return elapsed;
}

/* =============================================================
 * UTILITAS
 * ============================================================= */

/* Cari indeks gudang dengan skor terkecil */
int cari_terbaik(float *skor, int n) {
    int idx = 0;
    for (int i = 1; i < n; i++)
        if (skor[i] < skor[idx]) idx = i;
    return idx;
}

/* Tampilkan top-N gudang (pakai bubble sort sederhana untuk top-N) */
void cetak_top_n(float *skor, int n, int top_n) {
    int idx[MAX_GUDANG];
    for (int i = 0; i < n; i++) idx[i] = i;

    /* Bubble sort — hanya sort sebanyak top_n iterasi */
    for (int i = 0; i < top_n && i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (skor[idx[j]] < skor[idx[i]]) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
        }
        printf("  Rank %-3d | %-8s | lat=%8.4f | lon=%9.4f | "
               "kapasitas=%6d | skor=%.2f\n",
               i + 1,
               g_id[idx[i]],
               g_lat[idx[i]],
               g_lon[idx[i]],
               g_kap[idx[i]],
               skor[idx[i]]);
    }
}

/* Verifikasi bahwa dua array skor hasil dua metode konsisten */
void cek_konsistensi(float *skor_a, float *skor_b, const char *nama_b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(skor_a[i] - skor_b[i]) / (fabsf(skor_a[i]) + 1e-6f);
        if (err > max_err) max_err = err;
    }
    printf("  Konsistensi Sequential vs %s: max error = %.4f%%  %s\n",
           nama_b,
           max_err * 100.0f,
           max_err < 0.01f ? "[OK]" : "[PERLU DICEK]");
}

/* =============================================================
 * MAIN
 * ============================================================= */
int main(int argc, char *argv[]) {
    const char *path_bantuan = (argc > 1) ? argv[1] : "bantuan_sosial.csv";
    const char *path_gudang  = (argc > 2) ? argv[2] : "kandidat_gudang.csv";

    printf("=============================================================\n");
    printf("  Brute-Force Euclidean — Pencarian Gudang Strategis\n");
    printf("  Sequential | OpenMP | OpenCL\n");
    printf("=============================================================\n\n");

    /* ─── Muat data ─── */
    printf("[1] Memuat data...\n");
    if (baca_bantuan(path_bantuan) < 0) return 1;
    if (baca_gudang(path_gudang)   < 0) return 1;
    printf("    Penerima bantuan : %d titik\n",    n_bantuan);
    printf("    Kandidat gudang  : %d lokasi\n",   n_gudang);
    printf("    Total operasi    : %lld pasang jarak\n\n",
           (long long)n_bantuan * n_gudang);

    double t_seq, t_omp, t_ocl;

    /* ─── Sequential ─── */
    printf("[2] Sequential...\n");
    t_seq = hitung_sequential();
    int best_seq = cari_terbaik(skor_seq, n_gudang);
    printf("    Waktu          : %.4f detik\n", t_seq);
    printf("    Gudang terbaik : %s (skor = %.2f)\n\n",
           g_id[best_seq], skor_seq[best_seq]);

    /* ─── OpenMP ─── */
    printf("[3] OpenMP (%d thread tersedia)...\n", omp_get_max_threads());
    t_omp = hitung_openmp();
    int best_omp = cari_terbaik(skor_omp, n_gudang);
    printf("    Waktu          : %.4f detik\n", t_omp);
    printf("    Speedup        : %.2fx dari sequential\n", t_seq / t_omp);
    printf("    Gudang terbaik : %s (skor = %.2f)\n\n",
           g_id[best_omp], skor_omp[best_omp]);
    cek_konsistensi(skor_seq, skor_omp, "OpenMP", n_gudang);
    printf("\n");

    /* ─── OpenCL ─── */
    printf("[4] OpenCL...\n");
    t_ocl = hitung_opencl();
    if (t_ocl > 0.0) {
        int best_ocl = cari_terbaik(skor_ocl, n_gudang);
        printf("    Waktu          : %.4f detik\n", t_ocl);
        printf("    Speedup        : %.2fx dari sequential\n", t_seq / t_ocl);
        printf("    Gudang terbaik : %s (skor = %.2f)\n\n",
               g_id[best_ocl], skor_ocl[best_ocl]);
        cek_konsistensi(skor_seq, skor_ocl, "OpenCL", n_gudang);
    } else {
        printf("    OpenCL tidak tersedia. Coba: !apt-get install pocl-opencl-icd\n\n");
    }

    /* ─── Rangkuman benchmark ─── */
    printf("\n=============================================================\n");
    printf("  BENCHMARK SUMMARY\n");
    printf("=============================================================\n");
    printf("  %-22s %12s %12s\n", "Metode", "Waktu (s)", "Speedup");
    printf("  %-22s %12.4f %12s\n",  "Sequential",  t_seq, "1.00x (baseline)");
    printf("  %-22s %12.4f %11.2fx\n", "OpenMP (CPU)",  t_omp, t_seq / t_omp);
    if (t_ocl > 0.0)
        printf("  %-22s %12.4f %11.2fx\n", "OpenCL (GPU)", t_ocl, t_seq / t_ocl);
    else
        printf("  %-22s %12s %12s\n", "OpenCL (GPU)", "N/A", "N/A");

    /* ─── Top N gudang terbaik ─── */
    float *skor_final = (t_ocl > 0.0) ? skor_ocl : skor_seq;
    printf("\n  TOP %d GUDANG PALING STRATEGIS:\n", TOP_N);
    printf("  (skor kecil = dekat banyak penerima berprioritas tinggi)\n\n");
    cetak_top_n(skor_final, n_gudang, TOP_N);

    printf("\n[Selesai]\n");
    return 0;
}
