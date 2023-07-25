/*
* This is a class that holds the methods used for the epiblaster GPU Kernel
* Author: Lance Hartman
* Date: 7/20/2023
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.cuh"

#include <algorithm>

/**
* Matrix helper method cuz I couldn't figure out how to properly overload the operator. . .
* @param first First matrix
* @param other Matrix being subtracted onto the first matrix
* @returns First matrix - Second matrix
*/
__device__ Matrix subtract_matrices(const Matrix& first, const Matrix& other) {
    if (first.width != other.width || first.height != other.height) {
        printf("Error: cannot subtract matrixes with different sizes");
        return Matrix{};
    }
    else {
        Matrix ret = { first.width, first.height, first.elements };
        for (int i = 0; i < first.height; i++) {
            for (int j = 0; j < first.width; j++) {
                ret.elements[i * ret.width + j] -= other.elements[i * other.width + j];
            }
        }
        return ret;
    }
}

/**
* Method to transpose a matrix
* @param A Matrix to get transposed
* @returns A transposed copy of the matrix
*/
__device__ Matrix transpose(Matrix A)
{
    Matrix ret;
    ret.width = A.height;
    ret.height = A.width;
    ret.elements = new double[sizeof(A.elements) / sizeof(A.elements[0])];
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            // M(row, col) = *(M.elements + row * M.width + col)
            ret.elements[j * ret.width + i] = A.elements[i * A.width + j];
        }
    }
    return ret;
}

/**
* Cross product method (sort of)
* This replicates the R crossprod function which actually returns one matrix transposed multiplied by another matrix
* @param A One of the matrixes we will be using in our cross product
* @param B The other matrix we will be using
* @returns A matrix representation of the cross product (in this case: transposed matrix multiplied by other matrix)
*/
__device__ Matrix cross_product(Matrix A, Matrix B)
{
    Matrix C;
    Matrix a_transposed = transpose(A); //Transposed copy of A

    //Now we need to multiply the two matrices together
    if (a_transposed.width != B.height) {
        printf("Matrix sizes are not compatable!");
        return;
    }

    C.width = a_transposed.height;
    C.height = B.width;
    C.elements = new double[a_transposed.height * B.width];

    //First for loop -> per a_transposed row
    for (int k = 0; k < a_transposed.height; k++) {
        //Second for loop -> per B column
        for (int i = 0; i < B.width; i++) {
            //This is where we should determine which variable we are adding into
            double ret_element = 0;
            //Third for loop -> per B row
            for (int j = 0; j < B.height; j++) {
                // j should be the same for both a_transposed and B (if sizes are compatable)
                // M(row, col) = *(M.elements + row * M.width + col)
                ret_element += a_transposed.elements[k * a_transposed.width + j] * B.elements[j * B.width + i];
            }
            C.elements[k * C.width + i] = ret_element;
        }
    }
    return C;
}

/**
* Method to calculate the chunk range
* @param idx
* @param n
* @param chunk
* @returns Range struct holding chunk range info
*/
__device__ Range ithChunk(int idx, int n, int chunk)
{
    int start = (idx - 1) * chunk;
    if (idx < 0 || start > n) return Range{0,0}; // Should not happen!
    else return Range{start, (int)__CUDA_RUNTIME_H__::fmin((double)(idx * chunk), (double)n)}; //Make sure no rounding
}

/**
* Method to scale a matrix according to the R "scale()" function
* KEEP IN MIND SCALING IN CPP IS DIFFERENT THAN R
* @param A Matrix to be scaled
* @returns The scaled matrix
*/
__device__ Matrix scale(Matrix A){
    Matrix scaled = {
        A.width,
        A.height,
        new double[sizeof(A.elements)]
    };
    for (int i = 0; i < scaled.height; i++) {
        for (int j = 0; j < scaled.width; j++) {
            scaled.elements[i * scaled.width + j] = (*(A.elements + i * A.width + j) - A.mean()) / A.standard_dev();
        }
    }
    return scaled;
}

/**
* Calculates the Pearson Correlation Coefficient of two matrices
* @param A The first of the two matrixes
* @param B The second of the two matrixes
* @returns The correlation matrix
*/
__device__ Matrix getcor(Matrix A, Matrix B)
{
    if (A.height != B.height) {
        printf("A and B have different row numbers!");
        return Matrix{};
    }
    else {
        Matrix Abar = scale(A);
        //TODO issue with code below:
        for (int i = 0; i < Abar.height; i++) {
            for (int j = 0; j < Abar.width; j++) {
                Abar.elements[i * Abar.width + j] = Abar.elements[i * Abar.width + j]
                                                    * -1 * __CUDA_RUNTIME_H__::sqrt((double)(1.0 / (Abar.height - 1)));
            }
        }

        Matrix Bbar = scale(B);
        for (int i = 0; i < Bbar.height; i++) {
            for (int j = 0; j < Bbar.width; j++) {
                Bbar.elements[i * Bbar.width + j] = *(Bbar.elements + i * Abar.width + j)
                                                    * -1 * __CUDA_RUNTIME_H__::sqrt((double)(1.0 / (Bbar.height - 1)));
            }
        }
        return cross_product(Abar, Bbar);
    }
}

/**
Matrix A -> Transpose
Matrix B -> Normal
Multiply them together

      ztest <- (getcor(A = as.matrix(case[, ithChunk(as.numeric(i), nSNP, chunk), drop = FALSE]),
                       B = as.matrix(case[, ithChunk(as.numeric(j), nSNP, chunk), drop = FALSE]),
                       method = "pearson")
                -
                  getcor(A = as.matrix(control[, ithChunk(as.numeric(i), nSNP, chunk), drop = FALSE]),
                         B = as.matrix(control[, ithChunk(as.numeric(j), nSNP, chunk), drop = FALSE]),
                         method = "pearson") )  /  sd_tot
      index <- which(abs(ztest) >= zthres, arr.ind = TRUE)

      ifelse(i==j,
             WriteSnpPairs_sym,
             WriteSnpPairs)(Zmatrix = ztest, indexArr = index,
                            outfile = OUT)

*/
__global__ void EpiScanKernel(Matrix A, Matrix B) {
    printf("-----------KERNEL ACTIVATED-------------\n");
    A.print_matrix();
    printf("\n");
    B.print_matrix();
    printf("\n");

    /*
    Matrix C = cross_product(A, B);

    C.print_matrix();

    Matrix scaled = scale(A);
    scaled.print_matrix();

    Matrix corr = getcor(A, B);
    corr.print_matrix();
    */
    printf("-----------KERNEL FINISHED-------------\n");
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE************
cudaError_t EpiScan(const Matrix A, const Matrix B) {
    Matrix d_A = {};
    Matrix d_B = {};
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Load A to device memory
    size_t size = A.width * A.height * sizeof(double); //Calculate the total amount of memory to allocate for matrix A
    d_A.width = A.width;
    d_A.height = A.height;
    cudaStatus = cudaMalloc(&d_A.elements, size); //Allocate the data on the CUDA device
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_A.elements, A.elements, size,
        cudaMemcpyHostToDevice); //Copy the memory stored in the Matrix struct into the allocated memory
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        goto Error;
    }

    // Load B to device memory
    size = B.width * B.height * sizeof(double); //Calculate the total amount of memory to allocate for matrix A
    d_B.width = B.width;
    d_B.height = B.height;
    cudaStatus = cudaMalloc(&d_B.elements, size); //Allocate the data on the CUDA device
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_B.elements, B.elements, size,
        cudaMemcpyHostToDevice); //Copy the memory stored in the Matrix struct into the allocated memory
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        goto Error;
    }

    // Invoke kernel

    //struct cudaDeviceProp properties;
    //cudaGetDeviceProperties(&properties, device);
    //cout << "using " << properties.multiProcessorCount << " multiprocessors" << endl;
    //cout << "max threads per processor: " << properties.maxThreadsPerMultiProcessor << endl;


    dim3 dimBlock(16, 16);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    //EpiScanKernel <<<dimGrid, dimBlock >>>(&d_A, &d_B);
    EpiScanKernel <<<1, 1 >>> (d_A, d_B);

    cudaDeviceSynchronize();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "EpiScanKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

Error:
    return cudaStatus;
}
