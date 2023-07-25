#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

typedef struct Matrix {
    int width;
    int height;
    double* elements;

    __device__ void print_matrix();

    __device__ double mean();

    __device__ double standard_dev();
};

/**
* Struct to hold a range of values
* @param min Minimum range value
* @param max Maximum range value
*/
typedef struct {
    int min;
    int max;
} Range;

__device__ Matrix subtract_matrices(const Matrix& first, const Matrix& other);
__device__ Matrix transpose(Matrix A);
__device__ Matrix cross_product(Matrix A, Matrix B);
__device__ Range ithChunk(int idx, int n, int chunk);
__device__ Matrix scale(Matrix A);
__device__ Matrix getcor(Matrix A, Matrix B);
__global__ void EpiScanKernel(Matrix A, Matrix B);
__global__ void EpiScanKernel(Matrix A, Matrix B);
cudaError_t EpiScan(const Matrix A, const Matrix B);
