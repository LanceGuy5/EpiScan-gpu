/*
* This is a class that holds the methods used for the epiblaster GPU Kernel
* Author: Lance Hartman
* Date: 7/20/2023
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.cuh"

#include <algorithm>

__device__ void Matrix::print_matrix() {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f, ", *(elements + i * width + j));
        }
        printf("\n");
    }
    printf("\n");
}

__device__ double Matrix::mean() {
    double total = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            total += *(elements + i * width + j);
        }
    }
    return total / (width * height);
}

__device__ double Matrix::standard_dev() {
    double total = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            total += __CUDA_RUNTIME_API_H__::pow((*(elements + i * width + j) - mean()), 2);
        }
    }
    return sqrt((double)total / ((width * height) - 1));
}

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
__global__ void EpiScanKernel(Matrix genotype_data, 
                              Matrix phenotype_data, 
                              double* zpthres, 
                              int* chunksize
                             ) {
    printf("-----------KERNEL ACTIVATED-------------\n");
    
    //Check to make sure same number of cases for genotype and phenotype
    if (genotype_data.height != phenotype_data.height) {
        printf("A and B do not have the same number of elements. Please check your data!");
        return;
    }

    //Check to make sure that the chunksize isn't greater than the width of the matrix
    if (genotype_data.width > *chunksize) *chunksize = genotype_data.width;
    

    /*
    A.print_matrix();
    printf("\n");
    B.print_matrix();
    printf("\n");
    */
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
cudaError_t EpiScan(const Matrix A, 
                    const Matrix B, 
                    const double zpthres, 
                    const int chunksize
                   ) {
    Matrix d_A = {};
    Matrix d_B = {};
    double* d_zpthres;
    int* d_chunksize;
    cudaError_t cudaStatus;

    printf("EpiScan called!\n");

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
        printf("d_A cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_A.elements, A.elements, size,
        cudaMemcpyHostToDevice); //Copy the memory stored in the Matrix struct into the allocated memory
    if (cudaStatus != cudaSuccess) {
        printf("d_A cudaMemcpy failed!");
        goto Error;
    }

    // Load B to device memory
    size = B.width * B.height * sizeof(double); //Calculate the total amount of memory to allocate for matrix A
    d_B.width = B.width;
    d_B.height = B.height;
    cudaStatus = cudaMalloc(&d_B.elements, size); //Allocate the data on the CUDA device
    if (cudaStatus != cudaSuccess) {
        printf("d_B cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_B.elements, B.elements, size,
        cudaMemcpyHostToDevice); //Copy the memory stored in the Matrix struct into the allocated memory
    if (cudaStatus != cudaSuccess) {
        printf("d_B cudaMemcpy failed!");
        goto Error;
    }

    //Load zpthres to memory device
    cudaStatus = cudaMalloc((void**) & d_zpthres, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        printf("d_zpthres cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_zpthres, &zpthres, sizeof(double),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("d_zpthres cudaMemcpy failed!");
        goto Error;
    }

    //Load chunksize to memory device
    cudaStatus = cudaMalloc((void**)&d_chunksize, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("d_chunksize cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_chunksize, &chunksize, sizeof(int),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("d_chunksize cudaMemcpy failed!");
        goto Error;
    }

    printf("Memory Allocated!\n");

    // Invoke kernel

    //struct cudaDeviceProp properties;
    //cudaGetDeviceProperties(&properties, device);
    //cout << "using " << properties.multiProcessorCount << " multiprocessors" << endl;
    //cout << "max threads per processor: " << properties.maxThreadsPerMultiProcessor << endl;

    dim3 dimBlock(16, 16);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    EpiScanKernel <<<dimGrid, dimBlock>>> (d_A, d_B, d_zpthres, d_chunksize);

    cudaDeviceSynchronize();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "EpiScanKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaFree(&d_A);
    cudaFree(&d_B);
    cudaFree(d_zpthres);
    cudaFree(d_chunksize);

Error:
    cudaFree(&d_A);
    cudaFree(&d_B);
    cudaFree(d_zpthres);
    cudaFree(d_chunksize);

    return cudaStatus;
}
