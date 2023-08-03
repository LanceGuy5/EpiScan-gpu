#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#ifndef MAX_HEADER_LENGTH
    #define MAX_HEADER_LENGTH 25
#endif
#ifndef CHUNK_SZE
    #define CHUNK_SIZE 100
#endif
#ifndef DATA_HEIGHT
    #define DATA_HEIGHT 713
#endif
#ifndef TEST_PATH
    #define TEST_PATH "C:\\Users\\lance\\Desktop\\data\\data\\ALVM_imp_maf20perc_w_Target.csv"
#endif
#ifndef TEMP_FILE
    #define TEMP_FILE "C:\\Users\\lance\\Desktop\\data\\results\\alvm_results_temp.txt"
#endif
#ifndef OUTPUT_FILE
    #define OUTPUT_FILE "C:\\Users\\lance\\Desktop\\data\\results\\alvm_results_gpu.txt"
#endif
#ifndef ZPTHRES
    #define ZPTHRES 0.01//3//1e-6 - If the threshold is low enough, it can't write to the file fast enough
#endif
#ifndef MAX_LABEL_SIZE
    #define MAX_LABEL_SIZE 25
#endif
#ifndef MAX_COLS
    #define MAX_COLS 20000
#endif

/**
* Struct to hold a range of values
* @param min Minimum range value
* @param max Maximum range value
*/
typedef struct {
    const int min;
    const int max;

    __device__ int calc_dist();

    __device__ void print_range();
} Range;

typedef struct {
    const int width;
    const int height;
    double* elements;

    __device__ void print_matrix();
    __host__ void host_print_matrix();

    __device__ double mean(int column);
    __host__ double host_mean(int column);

    __device__ double standard_dev(double mean, int column);
    __host__ double host_standard_dev(double mean, int column);

} Matrix;

typedef struct {
    int id_one;
    int id_two;
    double z_score;
    double z_P;

    __device__ void print_entry();
} Entry;

__global__ void EpiScanKernel(
    Matrix d_case,
    Matrix d_control,
    double* d_zpthres,
    int* d_chunksize,
    int* d_geno_height,
    int* d_geno_width,
    int* d_pheno_height,
    int* d_pheno_width);
__global__ void ZTestKernel(
    int i,
    int* chunksize,
    Matrix control_mat,
    Matrix case_mat,
    double* zpthres,
    double sd_tot);
__device__ Matrix subtract_matrices(Matrix first, Matrix other);
__device__ Matrix transpose(Matrix A);
__device__ Matrix cross_product(Matrix A, Matrix B);
__device__ Range ithChunk(int idx, int n, int chunk);
__device__ Matrix getcor(Matrix A, Matrix B);
__device__ double ztoP(double zscore);

__host__ cudaError_t EpiScan(const Matrix A, const Matrix B, double zthres, int chunksize);
__host__ double qnorm(double p, double mean, double sd, bool lower_tail);
