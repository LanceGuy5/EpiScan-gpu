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
    int* d_pheno_width,
    int* d_flag);
__global__ void ZTestKernel(
    int i,
    int* block_dim,
    int* chunksize,
    Matrix control_mat,
    Matrix case_mat,
    double* zpthres,
    double sd_tot,
    int* d_flag);
__device__ Matrix subtract_matrices(const Matrix& first, const Matrix& other);
__device__ Matrix transpose(Matrix A);
__device__ Matrix cross_product(Matrix A, Matrix B);
__device__ Range ithChunk(int idx, int n, int chunk);
__device__ Matrix getcor(Matrix A, Matrix B);
__device__ double ztoP(double zscore);

cudaError_t EpiScan(const Matrix A, const Matrix B, double zpthres, int chunksize);
