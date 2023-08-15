/*
* This is a class that holds the methods used for the epiblaster GPU Kernel
* Author: Lance Hartman
* Date: 7/20/2023
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "epiblaster.cuh"

#include <algorithm>
#include <fstream>
#include "math.h"
#include <cmath>
#include <vector>
#include <thread>
#include <functional>
#ifdef _WIN32
    #include <windows.h>
#elif __linux__
    #include "pthread.h"
#endif
#ifndef PI
    #define PI 3.14159265358979323846
#endif

//Constant variables
__constant__ int n_SNP;

__device__ void Matrix::print_matrix() {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f, ", *(elements + i * width + j));
        }
        printf("\n");
    }
    printf("\n");
}

__host__ void Matrix::host_print_matrix() {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f, ", *(elements + i * width + j));
        }
        printf("\n");
    }
    printf("\n");
}

__device__ double Matrix::mean(int column) {
    double total = 0;
    for (int i = 0; i < height; i++) {
        //for (int j = 0; j < width; j++) {
        total += *(elements + i * width + column);
        //}
    }
    return total / height;
}

__host__ double Matrix::host_mean(int column) {
    double total = 0;
    for (int i = 0; i < height; i++) {
        //for (int j = 0; j < width; j++) {
        total += *(elements + i * width + column);
        //}
    }
    return total / height;
}

__device__ double Matrix::standard_dev(double mean, int column) {
    double total = 0;
    for (int i = 0; i < height; i++) {
        //for (int j = 0; j < width; j++) {
        total += pow((*(elements + i * width + column) - mean), 2);
        //}
    }
    return sqrt((double)total / (height - 1));
}

__host__ double Matrix::host_standard_dev(double mean, int column) {
    double total = 0;
    for (int i = 0; i < height; i++) {
        //for (int j = 0; j < width; j++) {
        total += pow((*(elements + i * width + column) - mean), 2);
        //}
    }
    return sqrt((double)total / (height - 1));
}

__device__ Matrix* sub_matrix(Matrix& main, Range height, Range width) {
    Matrix ret = {
        width.calc_dist(),
        height.calc_dist(),
        new double[width.calc_dist() * height.calc_dist()]
    };
    for (int i = height.min; i < height.max; i++) {
        for (int j = width.min; j < width.max; j++) {
            ret.elements[i * ret.width + j]
                = main.elements[i * main.width + j];
        }
    }
    return &ret;
}

__device__ int Range::calc_dist() {
    return max - min;
}

__host__ int Range::host_calc_dist() {
    return max - min;
}

__device__ void Range::print_range() {
    printf("%d, %d\n", min, max);
}

__host__ void Range::host_print_range() {
    printf("%d, %d\n", min, max);
}

__device__ void Entry::print_entry() {
    printf("%d, %d, %.15f, %.15f \n", id_one, id_two, z_score, z_P);
}

__host__ void Entry::host_print_entry() {
    printf("%d, %d, %.15f, %.15f \n", id_one, id_two, z_score, z_P);
}

/**
* Matrix helper method cuz I couldn't figure out how to properly overload the operator. . .
* @param first First matrix
* @param other Matrix being subtracted onto the first matrix
* @returns First matrix - Second matrix
*/
__device__ Matrix subtract_matrices(Matrix first, Matrix other) {
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
        //delete[] first.elements;
        //delete[] other.elements;
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
    //double* ret_elements = new double[sizeof(A.elements) / sizeof(A.elements[0])];
    double* ret_elements = new double[A.height * A.width];
    if (ret_elements == nullptr) printf("%d c_elements never made\n", threadIdx.x);
    Matrix ret{ A.height, A.width, ret_elements };
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
    Matrix a_transposed = transpose(A); //Transposed copy of A

    //Now we need to multiply the two matrices together
    if (a_transposed.width != B.height) {
        printf("Matrix sizes are not compatable!");
        return Matrix{};
    }

    double* C_elements = new double[a_transposed.height * B.width];

    if (C_elements == nullptr) {
        printf("%d c_elements never made\n", threadIdx.x);
        C_elements = new double[a_transposed.height * B.width];
    }

    Matrix C{ a_transposed.height, B.width, C_elements };

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

    delete[] a_transposed.elements;
    delete[] B.elements;

    return C;
}

__device__ void mat_divide(Matrix& matrix, double divisor) {
    for (int i = 0; i < matrix.height; i++) {
        for (int j = 0; j < matrix.width; j++) {
            *(matrix.elements + i * matrix.width + j)
                = *(matrix.elements + i * matrix.width + j) / divisor;
        }
    }
}

/**
* Method to calculate the chunk range
* @param idx
* @param n
* @param chunk
* @returns Range struct holding chunk range info
*/
__host__ Range ithChunk(int idx, int n, int chunk)
{
    int start = (idx - 1) * chunk;
    if (idx < 1 || start > n) return Range{ 0,0 }; // Should not happen!
    else return Range{ start, (int)fmin((double)(idx * chunk), (double)(n + 1)) - 1 }; //Make sure no rounding
}

/*
ZtoP <- function(z.score, ...){
  if(max(z.score)<=37.51929999999999765){
    return(2*pnorm(-abs(as.numeric(z.score))))
  }else{
    warning("There is some Z score value over the system length. After converting, the p-value is recorded as 1e-309.")
    pvals <- 2*pnorm(-abs(as.numeric(z.score)))
    pvals[pvals==0] <- 1e-309
    return(pvals)
  }
}
*/
__device__ double ztoP(double zscore) {
    if (zscore <= 37.51929999999999765) {
        return 1.0 + erf(-abs(zscore) / sqrt(2.0));
    }
    else {
        return 1e-309;
    }
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
        printf("A and B are incompatable in getcor calculation\n");
        return Matrix{};
    }
    //Matrix Abar = scale(A);
    double A_mean, A_stdev, B_mean, B_stdev;

    #pragma unroll
    for (int i = 0; i < A.width; i++) {
        A_mean = A.mean(i);
        A_stdev = A.standard_dev(A_mean, i);
        for (int j = 0; j < A.height; j++) {
            A.elements[j * A.width + i] = (*(A.elements + j * A.width + i) - A_mean) / A_stdev;
        }
    }
    #pragma unroll
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            A.elements[i * A.width + j] = A.elements[i * A.width + j]
                * sqrt((double)(1.0 / (A.height - 1)));
        }
    }

    #pragma unroll
    for (int i = 0; i < B.width; i++) {
        B_mean = B.mean(i);
        B_stdev = B.standard_dev(B_mean, i);
        for (int j = 0; j < B.height; j++) {
            B.elements[j * B.width + i] = (*(B.elements + j * B.width + i) - B_mean) / B_stdev;
        }
    }
    #pragma unroll
    for (int i = 0; i < B.height; i++) {
        for (int j = 0; j < B.width; j++) {
            B.elements[i * B.width + j] = *(B.elements + i * B.width + j)
                * sqrt((double)(1.0 / (B.height - 1)));
        }
    }

    return cross_product(A, B);
}


__global__ void ZTestKernel(
    int* i,
    Range i_chunk,
    Range j_chunk,
    int* chunksize,
    Matrix d_A_case,
    Matrix d_B_case,
    Matrix d_A_control,
    Matrix d_B_control,
    Entry* d_entries,
    double* zpthres,
    double* sd_tot) {

    //printf("Kernel started\n");

    //printf("[%d, %d], [%d, %d]\n", d_A_case.width, d_A_case.height, d_B_case.width, d_B_case.height);
    //printf("[%d, %d], [%d, %d]\n", d_A_control.width, d_A_control.height, d_B_control.width, d_B_control.height);

    Matrix z_test = subtract_matrices(
        getcor(d_A_case, d_B_case),
        getcor(d_A_control, d_B_control)
    );

    //printf("first section finished\n");

    //printf("%d, %d\n", z_test.height, z_test.width);
    //printf("%f\n", sd_tot);

    #pragma unroll
    for (int k = 0; k < z_test.height; k++) {
        for (int l = 0; l < z_test.width; l++) {
            z_test.elements[k * z_test.width + l] /= *sd_tot;
        }
    }

    //printf("%d: z_test actually finished\n", threadIdx.x);

    int entrySuccess = 0;

    //Actual search for interaction here
    #pragma unroll
    for (int k = 0; k < z_test.height; k++) {
        for (int l = 0; l < z_test.width; l++) {
            double d = *(z_test.elements + k * z_test.width + l);
            if (abs(d) >= *zpthres) {
                entrySuccess += 1;
                d_entries[k * z_test.height + l] = Entry
                {
                    k + i_chunk.min,
                    l + j_chunk.min,
                    d,
                    ztoP(d)
                };
                //d_entries[k * z_test.height + l]->print_entry();
            }
            else {
                d_entries[k * z_test.height + l] = Entry{};
            }
        }
    }

    delete[] z_test.elements;
    //printf("Kernel completed successfully!\n");

    //if(threadIdx.x == 0)
    //    printf("------------------Chunk %d finished--------------------\n", i);

}

/**
* What is run each time a thread is called in our subloop
*/
__host__ void individual_thread(int i, int j, int chunksize, Matrix& control_mat, Matrix& case_mat, double zpthres, double sd_tot, int SNP, cudaStream_t& currStream) {
    
    //printf("%lld started\n", std::this_thread::get_id());

    cudaError_t cudaStatus;

    int *d_i, *d_chunksize;
    double *d_zpthres, *d_sd_tot;

    Range i_chunk = ithChunk(i, SNP, chunksize);
    Range j_chunk = ithChunk(j + i, SNP, chunksize);

    double* A_chunk_case_data = new double[i_chunk.host_calc_dist() * case_mat.height];
    double* B_chunk_case_data = new double[j_chunk.host_calc_dist() * case_mat.height];
    double* A_chunk_control_data = new double[i_chunk.host_calc_dist() * control_mat.height];
    double* B_chunk_control_data = new double[j_chunk.host_calc_dist() * control_mat.height];

    if (A_chunk_case_data == nullptr || B_chunk_case_data == nullptr || A_chunk_control_data == nullptr || B_chunk_control_data == nullptr)
        printf("Something went wrong with the data initialization.");

    //Feed allocated arrays into matrix variables for ease of access
    Matrix A_chunk_case{ i_chunk.host_calc_dist(), case_mat.height, A_chunk_case_data };
    Matrix B_chunk_case{ j_chunk.host_calc_dist(), case_mat.height, B_chunk_case_data };
    Matrix A_chunk_control{ i_chunk.host_calc_dist(), control_mat.height, A_chunk_control_data };
    Matrix B_chunk_control{ j_chunk.host_calc_dist(), control_mat.height, B_chunk_control_data };

    //Writing data to arrays here
    for (int k = 0; k < A_chunk_case.height; k++) {
        for (int j = 0; j < A_chunk_case.width; j++) {
            A_chunk_case.elements[k * A_chunk_case.width + j]
                = case_mat.elements[k * A_chunk_case.width + j + i_chunk.min];
        }
    }

    for (int k = 0; k < B_chunk_case.height; k++) {
        for (int j = 0; j < B_chunk_case.width; j++) {
            B_chunk_case.elements[k * B_chunk_case.width + j]
                = case_mat.elements[k * B_chunk_case.width + j + j_chunk.min];
        }
    }

    for (int k = 0; k < A_chunk_control.height; k++) {
        for (int j = 0; j < A_chunk_control.width; j++) {
            A_chunk_control.elements[k * A_chunk_control.width + j]
                = control_mat.elements[k * A_chunk_control.width + j + i_chunk.min];
        }
    }

    for (int k = 0; k < B_chunk_control.height; k++) {
        for (int j = 0; j < B_chunk_control.width; j++) {
            B_chunk_control.elements[k * B_chunk_control.width + j]
                = control_mat.elements[k * B_chunk_control.width + j + j_chunk.min];
        }
    }

    cudaStatus = cudaMallocAsync(&d_i, sizeof(int), currStream);
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_i malloc\n");
        return;
    }
    //cudaStreamAttachMemAsync(currStream, d_i);
    
    cudaStatus = cudaMemcpyAsync(d_i, &i, sizeof(int), cudaMemcpyHostToDevice, currStream);
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_i memcpy\n");
        return;
    }

    cudaStatus = cudaMallocAsync(&d_chunksize, sizeof(int), currStream);
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_chunksize malloc\n");
        return;
    }
    //cudaStreamAttachMemAsync(currStream, d_chunksize);

    cudaStatus = cudaMemcpyAsync(d_chunksize, &chunksize, sizeof(int), cudaMemcpyHostToDevice, currStream);
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_chunksize memcpy\n");
        return;
    }

    cudaStatus = cudaMallocAsync(&d_zpthres, sizeof(double), currStream);
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_zpthres malloc\n");
        return;
    }
    //cudaStreamAttachMemAsync(currStream, d_zpthres);

    cudaStatus = cudaMemcpyAsync(d_zpthres, &zpthres, sizeof(double), cudaMemcpyHostToDevice, currStream);
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_zpthres memcpy\n");
        return;
    }

    cudaStatus = cudaMallocAsync(&d_sd_tot, sizeof(double), currStream);
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_sd_tot malloc\n");
        return;
    }
    //cudaStreamAttachMemAsync(currStream, d_sd_tot);

    cudaStatus = cudaMemcpyAsync(d_sd_tot, &sd_tot, sizeof(double), cudaMemcpyHostToDevice, currStream);
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_sd_tot memcpy\n");
        return;
    }
    //------------------------------------

    // Load all matrices to device memory
    Matrix d_A_case = { A_chunk_case.width, A_chunk_case.height, new double[A_chunk_case.width * A_chunk_case.height] };
    size_t size = A_chunk_case.width * A_chunk_case.height * sizeof(double); //Calculate the total amount of memory to allocate for matrix A
    cudaStatus = cudaMallocAsync(&d_A_case.elements, size, currStream); //Allocate the data on the CUDA device
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_A_case malloc\n");
        return;
    }
    //cudaStreamAttachMemAsync(currStream, d_A_case.elements);

    cudaStatus = cudaMemcpyAsync(d_A_case.elements, A_chunk_case.elements, size,
        cudaMemcpyHostToDevice, currStream); //Copy the memory stored in the Matrix struct into the allocated memory
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_A_case memcpy\n");
        return;
    }

    Matrix d_B_case = { B_chunk_case.width, B_chunk_case.height, new double[B_chunk_case.width * B_chunk_case.height] };
    size = B_chunk_case.width * B_chunk_case.height * sizeof(double); //Calculate the total amount of memory to allocate for matrix A
    cudaStatus = cudaMallocAsync(&d_B_case.elements, size, currStream); //Allocate the data on the CUDA device
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_B_case malloc\n");
        return;
    }
    //cudaStreamAttachMemAsync(currStream, d_B_case.elements);

    cudaStatus = cudaMemcpyAsync(d_B_case.elements, B_chunk_case.elements, size,
        cudaMemcpyHostToDevice, currStream); //Copy the memory stored in the Matrix struct into the allocated memory
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_B_case memcpy\n");
        return;
    }

    Matrix d_A_control = { A_chunk_control.width, A_chunk_control.height, new double[A_chunk_control.width * A_chunk_control.height] };
    size = A_chunk_control.width * A_chunk_control.height * sizeof(double); //Calculate the total amount of memory to allocate for matrix A
    cudaStatus = cudaMallocAsync(&d_A_control.elements, size, currStream); //Allocate the data on the CUDA device
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_A_control malloc\n");
        return;
    }
    //cudaStreamAttachMemAsync(currStream, d_A_control.elements);

    cudaStatus = cudaMemcpyAsync(d_A_control.elements, A_chunk_control.elements, size,
        cudaMemcpyHostToDevice, currStream); //Copy the memory stored in the Matrix struct into the allocated memory
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_A_control memcpy\n");
        return;
    }

    Matrix d_B_control = { B_chunk_control.width, B_chunk_control.height, new double[B_chunk_control.width * B_chunk_control.height] };
    size = B_chunk_control.width * B_chunk_control.height * sizeof(double); //Calculate the total amount of memory to allocate for matrix A
    cudaStatus = cudaMallocAsync(&d_B_control.elements, size, currStream); //Allocate the data on the CUDA device
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_B_control malloc\n");
        return;
    }
    //cudaStreamAttachMemAsync(currStream, d_B_control.elements);

    cudaStatus = cudaMemcpyAsync(d_B_control.elements, B_chunk_control.elements, size,
        cudaMemcpyHostToDevice, currStream); //Copy the memory stored in the Matrix struct into the allocated memory
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_B_control memcpy\n");
        return;
    }

    //Allocate some memory for returning Entries
    Entry* entries = new Entry[CHUNK_SIZE * CHUNK_SIZE];

    Entry* d_entries;
    cudaStatus = cudaMallocAsync((void**)&d_entries, sizeof(Entry) * CHUNK_SIZE * CHUNK_SIZE, currStream); //Allocate the data on the CUDA device
    if (cudaStatus != cudaSuccess) {
        printf("Error w d_entries malloc\n");
        return;
    }
    //cudaStreamAttachMemAsync(currStream, d_entries);

    ZTestKernel <<<1, 1, 0, currStream>>> (
        d_i,
        i_chunk,
        j_chunk,
        d_chunksize,
        d_A_case,
        d_B_case,
        d_A_control,
        d_B_control,
        d_entries,
        d_zpthres,
        d_sd_tot
    );

    //printf("Kernel exited\n");

    cudaMemcpyAsync(entries, d_entries, sizeof(Entry) * CHUNK_SIZE * CHUNK_SIZE,
        cudaMemcpyDeviceToHost, currStream); //Copy the memory stored in the Matrix struct into the allocated memory

    //cudaStreamSynchronize(currStream);

    //TODO WRITE ENTRIES TO FILE
    for (int i = 0; i < CHUNK_SIZE; i++) {
        for (int j = 0; j < CHUNK_SIZE; j++) {
            Entry curr = entries[i * CHUNK_SIZE + j];
            if (curr.z_P > 0.00001) { //Assumes this is not an empty Entry
                //curr.host_print_entry();
            }
        }
    }

    cudaFreeAsync(d_A_case.elements, currStream);
    cudaFreeAsync(d_B_case.elements, currStream);
    cudaFreeAsync(d_A_control.elements, currStream);
    cudaFreeAsync(d_B_control.elements, currStream);
    cudaFreeAsync(d_entries, currStream);
    cudaFreeAsync(d_i, currStream);
    cudaFreeAsync(d_chunksize, currStream);
    cudaFreeAsync(d_zpthres, currStream);
    cudaFreeAsync(d_sd_tot, currStream);
    
    delete[] A_chunk_case_data;
    delete[] B_chunk_case_data;
    delete[] A_chunk_control_data;
    delete[] B_chunk_control_data;
    delete[] entries;

    //printf("%lld finished\n", std::this_thread::get_id());
}

__host__ double qnorm(double p, double mean, double sd, bool lower_tail) {
    // Check for invalid inputs
    if (p <= 0 || p >= 1) {
        printf("Error: Probability must be between 0 and 1 (exclusive).\n");
        return NAN; // Return NaN for invalid inputs
    }

    // Calculate quantile for standard normal distribution (mean = 0, sd = 1)
    double z;
    if (lower_tail) {
        z = -std::sqrt(2.0) * erfcinv(2 * p); // fake error :(
    }
    else {
        z = std::sqrt(2.0) * erfcinv(2 * (1 - p));
    }

    return (double)(mean + z * sd);
}

__host__ cudaError_t EpiScan(Matrix genotype_data,
    Matrix phenotype_data,
    double zthres,
    const int chunksize) {
    double* d_zpthres;
    int* d_chunksize;

    cudaError_t cudaStatus;

    printf("EpiScan called!\n");

    //Check to make sure same number of cases for genotype and phenotype
    if (genotype_data.height != phenotype_data.height) {
        printf("A and B do not have the same number of elements. Please check your data!");
        goto Error;
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    printf("Refactoring zpthres\n");
    double zpthres = std::abs(qnorm(zthres, 0.0, 1.0, true));
    printf("%.15f\n", zpthres);

    //Setting size of the heap
    size_t heapSizeInBytes = 3000000000; //This is a *guestimate*
    size_t threadPrintFLimit = 1048576 * 3;

    // Set the heap size limit
    cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSizeInBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSetLimit failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaThreadSetLimit(cudaLimitPrintfFifoSize, threadPrintFLimit);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSetLimit failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    size_t freeMemory, totalMemory, printBufferSize;
    cudaStatus = cudaMemGetInfo(&freeMemory, &totalMemory);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed! %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    printf("Total GPU Memory: %zu bytes\n", totalMemory);
    printf("Free GPU Memory: %zu bytes\n", freeMemory);

    cudaStatus = cudaDeviceGetLimit(&printBufferSize, cudaLimitPrintfFifoSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaPrintBufferSize failed! %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    printf("printf buffer size: %zu bytes\n", printBufferSize);

    //Load zpthres to memory device
    cudaStatus = cudaMalloc((void**)&d_zpthres, sizeof(double));
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

    //Copy the value of geno_width to __constant__ memory
    cudaMemcpyToSymbol(n_SNP, &genotype_data.width, sizeof(int)); //Not real error

    //TODO ABOVE DEFINITELY NEEDED

    /*
    printf("Redefining output stream - output during kernels can be found in outfile\n");

    FILE* outputFile = freopen(TEMP_FILE, "w", stdout);
    if (outputFile == nullptr) {
        printf("Error opening file!\n");
        goto Error;
    }
    else {
        if (setvbuf(stdout, nullptr, _IONBF, 0) != 0) {
            printf("Error setting unbuffered mode for stdout!\n");
            goto Error;
        }
        cudaFree(0);
    }
    */

    //Determine the case and control matrices BEFORE scaling phenotype_data (easier)
    int control_count = 0, case_count = 0;
    for (int i = 0; i < phenotype_data.height; i++) {
        if (phenotype_data.elements[i] == 0.0) control_count++;
        else case_count++;
    }

    //Scaling data - different than expected
    double geno_mean = 0.0, geno_stdev = 0.0, pheno_mean = 0.0, pheno_stdev = 0.0;

    for (int i = 0; i < genotype_data.width; i++) {
        geno_mean = genotype_data.host_mean(i);
        geno_stdev = genotype_data.host_standard_dev(geno_mean, i);
        for (int j = 0; j < genotype_data.height; j++) {
            genotype_data.elements[j * genotype_data.width + i]
                = (*(genotype_data.elements + j * genotype_data.width + i) - geno_mean) / geno_stdev;
        }
    }

    for (int i = 0; i < phenotype_data.width; i++) {
        pheno_mean = phenotype_data.host_mean(i);
        pheno_stdev = phenotype_data.host_standard_dev(pheno_mean, i);
        for (int j = 0; j < phenotype_data.height; j++) {
            phenotype_data.elements[j * phenotype_data.width + i]
                = (*(phenotype_data.elements + j * phenotype_data.width + i) - pheno_mean) / pheno_stdev;
        }
    }

    Matrix control_mat{ genotype_data.width, control_count, new double[control_count * genotype_data.width] };
    Matrix case_mat{ genotype_data.width, case_count, new double[case_count * genotype_data.width] };

    // M(row, col) = *(M.elements + row * M.width + col)
    control_count = (case_count = 0);
    for (int i = 0; i < phenotype_data.height; i++) {
        if (phenotype_data.elements[i] <= 0.0) {
            for (int j = 0; j < genotype_data.width; j++) {
                control_mat.elements[control_count * control_mat.width + j]
                    = genotype_data.elements[i * genotype_data.width + j];
            }
            control_count++;
        }
        else {
            for (int j = 0; j < genotype_data.width; j++) {
                *(case_mat.elements + case_count * case_mat.width + j)
                    = *(genotype_data.elements + i * genotype_data.width + j);
            }
            case_count++;
        }
    }

    printf("Case and control generated\n");

    int n_splits = 0;

    //Check to make sure that the chunksize isn't greater than the width of the matrix
    if (genotype_data.width < chunksize) n_splits = (int)ceilf((float)genotype_data.width / (float)genotype_data.width);
    else n_splits = (int)ceilf((float)genotype_data.width / (float)chunksize);

    printf("Preparing %d chunk loops...\n", n_splits);

    double sd_tot = sqrt(
        (1.0 / (double)(control_mat.height - 1)) + (1.0 / (double)(case_mat.height - 1))
    );

    //Load chunksize to device memory
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

    //Check for sd_tot - sd_tot IS GOOD
    //printf("%.15f\n", sd_tot);

    //Check to make sure dims are right - they are
    //printf("%d, %d\n", control_mat.height, case_mat.height);
    //printf("%d, %d\n", control_mat.width, case_mat.width);

    //Here is where the normal cluster code belongs -> time to divide into a different kernel :(
    //Chunk calculation starts at 1
    printf("Main loop starting...\n");
    printf("Thread limit(?): %u\n", std::thread::hardware_concurrency());
    cudaStream_t* streams = new cudaStream_t[n_splits];
    for (int j = 0; j < n_splits; j++) {
        cudaStream_t currStream;
        cudaStreamCreate(&currStream);
        streams[j] = currStream;
    }
    size_t free, total; //for data checking
    for (int i = 1; i <= n_splits; i++) {
        printf("-------------Chunk %d-------------\n", i);
        Range curr_range = { i, n_splits };
        int thread_dim = (curr_range.max - curr_range.min) + 1; //Get rid of +1 in order to run with fixed way?
        std::vector<std::thread> t;
        for (int j = 0; j < thread_dim; j++) {
            std::thread curr(individual_thread, 
                             i, 
                             j, 
                             chunksize, 
                             std::ref(control_mat), 
                             std::ref(case_mat), 
                             zpthres, 
                             sd_tot,
                             genotype_data.width,
                             std::ref(streams[j]));
            t.push_back(move(curr));
        }
        for (int j = 0; j < thread_dim; j++) {
            t.at(j).join();
            cudaStreamSynchronize(streams[j]);
        }
        cudaMemGetInfo(&freeMemory, &totalMemory);
        printf("Total GPU Memory: %zu bytes\n", totalMemory);
        printf("Free GPU Memory: %zu bytes\n", freeMemory);
    }
    for (int j = 0; j < n_splits; j++) {
        cudaStreamDestroy(streams[j]);
    }
    delete[] streams;

    cudaDeviceSynchronize();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "EpiScanKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    //Remove phenotype and genotype data
    delete[] phenotype_data.elements;
    delete[] genotype_data.elements;
    delete[] case_mat.elements;
    delete[] control_mat.elements;

    cudaFree(d_zpthres);
    cudaFree(d_chunksize);

    //Fix handle
    /*
    #ifdef _WIN32
        SetStdHandle(STD_OUTPUT_HANDLE, hOut);
        CloseHandle(hFile);
    #elif __linux__
        //Add linux imports here
    #endif
    */
    fflush(stdout);
    freopen("CON", "w", stdout);
    printf("Output stream fixed - back to here.\n");

Error:
    delete[] phenotype_data.elements;
    delete[] genotype_data.elements;
    delete[] case_mat.elements;
    delete[] control_mat.elements;

    cudaFree(d_zpthres);
    cudaFree(d_chunksize);

    //Fix handle
    /*
    #ifdef _WIN32
        SetStdHandle(STD_OUTPUT_HANDLE, hOut);
        CloseHandle(hFile);
    #elif __linux__
        //Add linux imports here
    #endif
    */
    fflush(stdout);
    freopen("CON", "w", stdout);

    return cudaStatus;
}
