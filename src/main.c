/*
 * Program pemilihan gudang paling strategis.
 *
 * Compile:
 *   gcc -O2 -fopenmp main.c -o gudang_opencl -lOpenCL -lm
 *
 * Jalankan:
 *   ./gudang_opencl bantuan_sosial.csv kandidat_gudang.csv ranking_final.csv
 */

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_BANTUAN 55000
#define MAX_GUDANG 1100
#define MAX_LINE 256

typedef struct {
    float lat;
    float lon;
    float bobot;
} Bantuan;

typedef struct {
    char id[16];
    float lat;
    float lon;
    int kapasitas;
    float skor;
} Gudang;

static Bantuan bantuan[MAX_BANTUAN];
static Gudang gudang[MAX_GUDANG];
static int jumlah_bantuan = 0;
static int jumlah_gudang = 0;

static const char *opencl_kernel =
"__kernel void hitung_skor(\n"
"    __global const float *b_lat, __global const float *b_lon, __global const float *b_bobot,\n"
"    __global const float *g_lat, __global const float *g_lon, __global float *hasil,\n"
"    int jumlah_bantuan, int jumlah_gudang)\n"
"{\n"
"    int g = get_global_id(0);\n"
"    if (g >= jumlah_gudang) return;\n"
"\n"
"    float total = 0.0f;\n"
"    for (int i = 0; i < jumlah_bantuan; i++) {\n"
"        float dlat = g_lat[g] - b_lat[i];\n"
"        float dlon = g_lon[g] - b_lon[i];\n"
"        float jarak = sqrt(dlat * dlat + dlon * dlon);\n"
"        total += jarak * b_bobot[i];\n"
"    }\n"
"    hasil[g] = total;\n"
"}\n";

typedef struct {
    double waktu_ms;
    char nama_device[128];
} HasilOpenCL;

static double waktu_sekarang(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int baca_bantuan(const char *nama_file) {
    FILE *file = fopen(nama_file, "r");
    if (!file) {
        perror("Gagal membuka file bantuan");
        return 0;
    }

    char baris[MAX_LINE];
    fgets(baris, sizeof(baris), file); /* lewati header */

    while (jumlah_bantuan < MAX_BANTUAN && fgets(baris, sizeof(baris), file)) {
        char id[20];
        float lat, lon;
        int penerima, prioritas;

        if (sscanf(baris, "%19[^,],%f,%f,%d,%d", id, &lat, &lon, &penerima, &prioritas) == 5) {
            bantuan[jumlah_bantuan].lat = lat;
            bantuan[jumlah_bantuan].lon = lon;
            bantuan[jumlah_bantuan].bobot = (float)penerima * (float)prioritas;
            jumlah_bantuan++;
        }
    }

    fclose(file);
    return jumlah_bantuan;
}

static int baca_gudang(const char *nama_file) {
    FILE *file = fopen(nama_file, "r");
    if (!file) {
        perror("Gagal membuka file gudang");
        return 0;
    }

    char baris[MAX_LINE];
    fgets(baris, sizeof(baris), file); /* lewati header */

    while (jumlah_gudang < MAX_GUDANG && fgets(baris, sizeof(baris), file)) {
        char id[16];
        float lat, lon;
        int kapasitas;

        if (sscanf(baris, "%15[^,],%f,%f,%d", id, &lat, &lon, &kapasitas) == 4) {
            strcpy(gudang[jumlah_gudang].id, id);
            gudang[jumlah_gudang].lat = lat;
            gudang[jumlah_gudang].lon = lon;
            gudang[jumlah_gudang].kapasitas = kapasitas;
            gudang[jumlah_gudang].skor = 0.0f;
            jumlah_gudang++;
        }
    }

    fclose(file);
    return jumlah_gudang;
}

static float hitung_skor_satu_gudang(Gudang g) {
    float total = 0.0f;

    for (int i = 0; i < jumlah_bantuan; i++) {
        float dlat = g.lat - bantuan[i].lat;
        float dlon = g.lon - bantuan[i].lon;
        float jarak = sqrtf(dlat * dlat + dlon * dlon);

        total += jarak * bantuan[i].bobot;
    }

    return total;
}

static void hitung_sequential(float hasil[]) {
    for (int i = 0; i < jumlah_gudang; i++) {
        hasil[i] = hitung_skor_satu_gudang(gudang[i]);
    }
}

static void hitung_parallel(float hasil[]) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < jumlah_gudang; i++) {
        hasil[i] = hitung_skor_satu_gudang(gudang[i]);
    }
}

static double benchmark(void (*fungsi)(float[]), float hasil[]) {
    double mulai = waktu_sekarang();
    fungsi(hasil);
    return (waktu_sekarang() - mulai) * 1000.0;
}

static double selisih_maksimum(float a[], float b[]) {
    double maksimum = 0.0;

    for (int i = 0; i < jumlah_gudang; i++) {
        double selisih = fabs((double)a[i] - (double)b[i]);
        if (selisih > maksimum) maksimum = selisih;
    }

    return maksimum;
}

static int pilih_device_opencl(cl_device_id *device, char nama_device[]) {
    cl_platform_id platform[8];
    cl_uint jumlah_platform = 0;

    if (clGetPlatformIDs(8, platform, &jumlah_platform) != CL_SUCCESS) return 0;

    for (cl_uint i = 0; i < jumlah_platform; i++) {
        if (clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, 1, device, NULL) == CL_SUCCESS ||
            clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_CPU, 1, device, NULL) == CL_SUCCESS) {
            clGetDeviceInfo(*device, CL_DEVICE_NAME, 128, nama_device, NULL);
            return 1;
        }
    }

    return 0;
}

static int hitung_opencl(float hasil[], HasilOpenCL *info) {
    cl_int err;
    cl_device_id device;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem b_lat = NULL, b_lon = NULL, b_bobot = NULL, g_lat = NULL, g_lon = NULL, d_hasil = NULL;
    int sukses = 0;

    float *bantuan_lat = malloc((size_t)jumlah_bantuan * sizeof(float));
    float *bantuan_lon = malloc((size_t)jumlah_bantuan * sizeof(float));
    float *bantuan_bobot = malloc((size_t)jumlah_bantuan * sizeof(float));
    float *gudang_lat = malloc((size_t)jumlah_gudang * sizeof(float));
    float *gudang_lon = malloc((size_t)jumlah_gudang * sizeof(float));
    if (!bantuan_lat || !bantuan_lon || !bantuan_bobot || !gudang_lat || !gudang_lon) goto selesai;

    for (int i = 0; i < jumlah_bantuan; i++) {
        bantuan_lat[i] = bantuan[i].lat;
        bantuan_lon[i] = bantuan[i].lon;
        bantuan_bobot[i] = bantuan[i].bobot;
    }
    for (int i = 0; i < jumlah_gudang; i++) {
        gudang_lat[i] = gudang[i].lat;
        gudang_lon[i] = gudang[i].lon;
    }

    if (!pilih_device_opencl(&device, info->nama_device)) goto selesai;

    double mulai = waktu_sekarang();
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) goto selesai;

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) goto selesai;

    program = clCreateProgramWithSource(context, 1, &opencl_kernel, NULL, &err);
    if (err != CL_SUCCESS) goto selesai;
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) goto selesai;

    kernel = clCreateKernel(program, "hitung_skor", &err);
    if (err != CL_SUCCESS) goto selesai;

    size_t bantuan_bytes = (size_t)jumlah_bantuan * sizeof(float);
    size_t gudang_bytes = (size_t)jumlah_gudang * sizeof(float);

    b_lat = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bantuan_bytes, bantuan_lat, &err);
    if (err != CL_SUCCESS) goto selesai;
    b_lon = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bantuan_bytes, bantuan_lon, &err);
    if (err != CL_SUCCESS) goto selesai;
    b_bobot = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bantuan_bytes, bantuan_bobot, &err);
    if (err != CL_SUCCESS) goto selesai;
    g_lat = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gudang_bytes, gudang_lat, &err);
    if (err != CL_SUCCESS) goto selesai;
    g_lon = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gudang_bytes, gudang_lon, &err);
    if (err != CL_SUCCESS) goto selesai;
    d_hasil = clCreateBuffer(context, CL_MEM_WRITE_ONLY, gudang_bytes, NULL, &err);
    if (err != CL_SUCCESS) goto selesai;

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_lat);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_lon);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_bobot);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &g_lat);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &g_lon);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_hasil);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &jumlah_bantuan);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &jumlah_gudang);
    if (err != CL_SUCCESS) goto selesai;

    size_t global = (size_t)jumlah_gudang;
    if (clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL) != CL_SUCCESS) goto selesai;
    if (clFinish(queue) != CL_SUCCESS) goto selesai;
    if (clEnqueueReadBuffer(queue, d_hasil, CL_TRUE, 0, gudang_bytes, hasil, 0, NULL, NULL) != CL_SUCCESS) goto selesai;

    info->waktu_ms = (waktu_sekarang() - mulai) * 1000.0;
    sukses = 1;

selesai:
    if (d_hasil) clReleaseMemObject(d_hasil);
    if (g_lon) clReleaseMemObject(g_lon);
    if (g_lat) clReleaseMemObject(g_lat);
    if (b_bobot) clReleaseMemObject(b_bobot);
    if (b_lon) clReleaseMemObject(b_lon);
    if (b_lat) clReleaseMemObject(b_lat);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    free(gudang_lon);
    free(gudang_lat);
    free(bantuan_bobot);
    free(bantuan_lon);
    free(bantuan_lat);
    return sukses;
}

static int bandingkan_skor(const void *a, const void *b) {
    const Gudang *g1 = (const Gudang *)a;
    const Gudang *g2 = (const Gudang *)b;

    if (g1->skor < g2->skor) return -1;
    if (g1->skor > g2->skor) return 1;
    return 0;
}

static void simpan_ranking(const char *nama_file) {
    FILE *file = fopen(nama_file, "w");
    if (!file) {
        perror("Gagal membuat file output");
        return;
    }

    Gudang ranking[MAX_GUDANG];
    memcpy(ranking, gudang, (size_t)jumlah_gudang * sizeof(Gudang));
    qsort(ranking, (size_t)jumlah_gudang, sizeof(Gudang), bandingkan_skor);

    fprintf(file, "Rank,ID_Gudang,Latitude,Longitude,Kapasitas,Skor_Strategis\n");
    for (int i = 0; i < jumlah_gudang; i++) {
        fprintf(file, "%d,%s,%.6f,%.6f,%d,%.4f\n",
                i + 1,
                ranking[i].id,
                ranking[i].lat,
                ranking[i].lon,
                ranking[i].kapasitas,
                ranking[i].skor);
    }

    fclose(file);
}

static void tampilkan_gudang_terbaik(void) {
    int terbaik = 0;

    for (int i = 1; i < jumlah_gudang; i++) {
        if (gudang[i].skor < gudang[terbaik].skor) {
            terbaik = i;
        }
    }

    printf("\nGudang paling strategis:\n");
    printf("  ID        : %s\n", gudang[terbaik].id);
    printf("  Latitude  : %.6f\n", gudang[terbaik].lat);
    printf("  Longitude : %.6f\n", gudang[terbaik].lon);
    printf("  Kapasitas : %d\n", gudang[terbaik].kapasitas);
    printf("  Skor      : %.4f\n", gudang[terbaik].skor);
}

int main(int argc, char *argv[]) {
    const char *file_bantuan = (argc > 1) ? argv[1] : "bantuan_sosial.csv";
    const char *file_gudang = (argc > 2) ? argv[2] : "kandidat_gudang.csv";
    const char *file_output = (argc > 3) ? argv[3] : "ranking_final.csv";

    printf("=== Pemilihan Gudang Strategis ===\n\n");

    if (!baca_bantuan(file_bantuan) || !baca_gudang(file_gudang)) {
        fprintf(stderr, "Data gagal dibaca atau file kosong.\n");
        return 1;
    }

    printf("Jumlah titik bantuan : %d\n", jumlah_bantuan);
    printf("Jumlah kandidat      : %d\n", jumlah_gudang);
    printf("Total perhitungan    : %lld jarak\n\n", (long long)jumlah_bantuan * jumlah_gudang);

    float skor_sequential[MAX_GUDANG];
    float skor_parallel[MAX_GUDANG];
    float skor_opencl[MAX_GUDANG];

    double waktu_seq = benchmark(hitung_sequential, skor_sequential);
    double waktu_par = benchmark(hitung_parallel, skor_parallel);
    HasilOpenCL opencl = {0.0, "-"};
    int opencl_tersedia = hitung_opencl(skor_opencl, &opencl);

    for (int i = 0; i < jumlah_gudang; i++) {
        gudang[i].skor = skor_parallel[i];
    }

    int thread = 1;
#ifdef _OPENMP
    thread = omp_get_max_threads();
#endif

    printf("Benchmark:\n");
    printf("  Sequential CPU : %.4f ms\n", waktu_seq);
    printf("  OpenMP CPU     : %.4f ms (%d thread), speedup %.2fx, selisih %.6f\n",
           waktu_par, thread, waktu_seq / waktu_par, selisih_maksimum(skor_sequential, skor_parallel));
#ifndef _OPENMP
    printf("  Catatan        : compile dengan -fopenmp agar OpenMP memakai banyak thread.\n");
#endif

    if (opencl_tersedia) {
        printf("  OpenCL         : %.4f ms (%s), speedup %.2fx, selisih %.6f\n",
               opencl.waktu_ms,
               opencl.nama_device,
               waktu_seq / opencl.waktu_ms,
               selisih_maksimum(skor_sequential, skor_opencl));
    } else {
        printf("  OpenCL         : tidak tersedia di komputer/runtime ini.\n");
    }

    tampilkan_gudang_terbaik();
    simpan_ranking(file_output);

    printf("\nRanking lengkap disimpan ke: %s\n", file_output);
    printf("Selesai.\n");

    return 0;
}
