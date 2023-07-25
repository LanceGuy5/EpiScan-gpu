#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

typedef struct Matrix {
    int width;
    int height;
    double* elements;

    __device__ void print_matrix() {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%f, ", *(elements + i * width + j));
            }
            printf("\n");
        }
        printf("\n");
    }

    __device__ double mean() {
        double total = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                total += *(elements + i * width + j);
            }
        }
        return total / (width * height);
    }

    __device__ double standard_dev() {
        double total = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                total += pow((*(elements + i * width + j) - mean()), 2);
            }
        }
        return sqrt((double)total / ((width * height) - 1));
    }
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
