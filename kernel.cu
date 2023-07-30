/*
* This is a class that holds the methods used for the epiblaster GPU Kernel
* Author: Lance Hartman
* Date: 7/20/2023
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.cuh"

#include <algorithm>

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

__device__ double Matrix::mean() {
    double total = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            total += *(elements + i * width + j);
        }
    }
    return total / (width * height);
}

__device__ double Matrix::standard_dev(double mean) {
    double total = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            total += pow((*(elements + i * width + j) - mean), 2);
        }
    }
    return sqrt((double)total / ((width * height) - 1));
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

__device__ void Range::print_range() {
    printf("%d, %d\n", min, max);
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
    //double* ret_elements = new double[sizeof(A.elements) / sizeof(A.elements[0])];
    double* ret_elements = new double[A.height * A.width];
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

    delete[] A.elements;

    //Now we need to multiply the two matrices together
    if (a_transposed.width != B.height) {
        printf("Matrix sizes are not compatable!");
        return Matrix{};
    }

    //double* C_elements = new double[a_transposed.height * B.width];
    double* C_elements = new double[a_transposed.height * B.width];

    Matrix C { a_transposed.height, B.width, C_elements };

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
__device__ Range ithChunk(int idx, int n, int chunk)
{
    int start = (idx - 1) * chunk;
    if (idx < 1 || start > n) return Range{0,0}; // Should not happen!
    else return Range{start, (int)fmin((double)(idx * chunk), (double)(n + 1))}; //Make sure no rounding
}

/**
* Method to scale a matrix according to the R "scale()" function
* KEEP IN MIND SCALING IN CPP IS DIFFERENT THAN R
* @param A Matrix to be scaled
* @returns The scaled matrix
*/
__device__ Matrix scale(Matrix A){
    double* ret_elements = (double*)malloc(A.width * A.height);
    Matrix ret = { A.width, A.height, ret_elements };
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            ret.elements[i * ret.width + j] = (*(A.elements + i * A.width + j) - A.mean()) / A.standard_dev(A.mean());
        }
    }
    return ret;
}

/**
* Calculates the Pearson Correlation Coefficient of two matrices
* @param A The first of the two matrixes
* @param B The second of the two matrixes
* @returns The correlation matrix
*/
__device__ Matrix getcor(Matrix A, Matrix B)
{
    //Matrix Abar = scale(A);
    double A_mean = A.mean();
    double A_stdev = A.standard_dev(A_mean);
    double B_mean = B.mean();
    double B_stdev = B.standard_dev(B_mean);

    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            A.elements[i * A.width + j] = (*(A.elements + i * A.width + j) - A_mean) / A_stdev;
        }
    }
    //TODO issue with code below:
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            A.elements[i * A.width + j] = A.elements[i * A.width + j]
                                                * -1 * __CUDA_RUNTIME_H__::sqrt((double)(1.0 / (A.height - 1)));
        }
    }

    //Matrix Bbar = scale(B);
    for (int i = 0; i < B.height; i++) {
        for (int j = 0; j < B.width; j++) {
            B.elements[i * B.width + j] = (*(B.elements + i * B.width + j) - B_mean) / B_stdev;
        }
    }
    for (int i = 0; i < B.height; i++) {
        for (int j = 0; j < B.width; j++) {
            B.elements[i * B.width + j] = *(B.elements + i * B.width + j)
                                                * -1 * __CUDA_RUNTIME_H__::sqrt((double)(1.0 / (B.height - 1)));
        }
    }

    return cross_product(A, B);
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
__global__ void ZTestKernel(int i, 
                            int* thread_dim,
                            int* chunksize,
                            Matrix control_mat, 
                            Matrix case_mat,
                            double* zpthres,
                            double* sd_tot,
                            int* d_flag) {

    //Find matrix ranges for analysis based on chunks
    //i = i, j = threadIdx.x

    Range i_chunk = ithChunk(i, n_SNP, *chunksize);
    Range j_chunk = ithChunk((int)threadIdx.x + 1, n_SNP, *chunksize);

    double* A_chunk_case_data = new double[i_chunk.calc_dist() * case_mat.height];
    double* B_chunk_case_data = new double[j_chunk.calc_dist() * case_mat.height];
    double* A_chunk_control_data = new double[i_chunk.calc_dist() * control_mat.height];
    double* B_chunk_control_data = new double[j_chunk.calc_dist() * control_mat.height];

    //Feed allocated arrays into matrix variables for ease of access
    Matrix A_chunk_case{ i_chunk.calc_dist(), case_mat.height, A_chunk_case_data };
    Matrix B_chunk_case{ j_chunk.calc_dist(), case_mat.height, B_chunk_case_data };
    Matrix A_chunk_control{ i_chunk.calc_dist(), control_mat.height, A_chunk_control_data };
    Matrix B_chunk_control{ j_chunk.calc_dist(), control_mat.height, B_chunk_control_data };
    
    //Writing data to arrays here
    for (int i = 0; i < A_chunk_case.height; i++) {
        for (int j = 0; j < A_chunk_case.width; j++) {
            A_chunk_case.elements[i * A_chunk_case.width + j]
                = case_mat.elements[i * A_chunk_case.width + j + i_chunk.min];
        }
    }

    for (int i = 0; i < B_chunk_case.height; i++) {
        for (int j = 0; j < B_chunk_case.width; j++) {
            B_chunk_case.elements[i * B_chunk_case.width + j]
                = case_mat.elements[i * B_chunk_case.width + j + j_chunk.min];
        }
    }

    for (int i = 0; i < A_chunk_control.height; i++) {
        for (int j = 0; j < A_chunk_control.width; j++) {
            A_chunk_control.elements[i * A_chunk_control.width + j]
                = case_mat.elements[i * A_chunk_control.width + j + i_chunk.min];
        }
    }

    for (int i = 0; i < B_chunk_control.height; i++) {
        for (int j = 0; j < B_chunk_control.width; j++) {
            B_chunk_control.elements[i * B_chunk_control.width + j]
                = case_mat.elements[i * B_chunk_control.width + j + j_chunk.min];
        }
    }

    if(threadIdx.x == 0)
        printf("%d bytes allocated in heap memory for this chunk\n", sizeof(double) * A_chunk_case.width * A_chunk_case.height * 4 * 167);

    //Need to fix some part of this method
    Matrix z_test = subtract_matrices(
        getcor(
            A_chunk_case,
            B_chunk_case
        ),
        getcor(
            A_chunk_control,
            B_chunk_control
        )
    );

    if(threadIdx.x == 0)
        z_test.print_matrix();

    printf("-----------------------------------\n");

    mat_divide(z_test, *sd_tot);

    if (threadIdx.x == 0)
        z_test.print_matrix();

    //printf("%d z_test performed\n", threadIdx.x);

    __syncthreads(); //Not real error

    //Freeing all data
    delete[] A_chunk_case_data;
    delete[] B_chunk_case_data;
    delete[] A_chunk_control_data;
    delete[] B_chunk_control_data;

    printf("Deleted chunk data\n");

    delete[] z_test.elements;

    printf("Deleted ztest data\n");

    __syncthreads(); //Not real error

    if (threadIdx.x == *thread_dim) {
        *d_flag = 1;
    }

}

/**
* Kernel to represent the main bulk of the cluster division and management for episcan
* It also takes care of exporting "entries" to shared memory (TODO)
* @param case_mat a matrix representing the case data
* @param control_mat a matrix representing the control data
* @param zpthres the z threshold for the z test kernel
* @param chunksize the size of chunks (process division)
* @geno_height the height of the geno data - num of cases
* @param geno_width the width of the geno data - num of features
* @param pheno_height the height of the pheno data - num of cases
* @param pheno_width the width of the pheno data - should be 1
*/
__global__ void EpiScanKernel(Matrix case_mat, 
                              Matrix control_mat, 
                              double* zpthres, 
                              int* chunksize,
                              int* geno_height,
                              int* geno_width,
                              int* pheno_height,
                              int* pheno_width,
                              int* d_flag) {
    printf("-----------KERNEL ACTIVATED-------------\n");
    
    //Check to make sure same number of cases for genotype and phenotype
    if (*geno_height != *pheno_height) {
        printf("A and B do not have the same number of elements. Please check your data!");
        return;
    }

    //Check to make sure that the chunksize isn't greater than the width of the matrix
    if (*geno_width < *chunksize) *chunksize = *geno_width;
    
    int n_splits = (int)ceilf((float)n_SNP / (float)*chunksize);

    printf("Preparing %d chunk loops...\n", n_splits);

    double sd_tot = __CUDA_RUNTIME_H__::sqrt(
        (1.0 / (double)(control_mat.height - 1)) + (1.0 / (double)(case_mat.height - 1))
    );

    //Check to make sure dims are right - they are
    //printf("%d, %d\n", control_mat.height, case_mat.height);
    //printf("%d, %d\n", control_mat.width, case_mat.width);

    //Here is where the normal cluster code belongs -> time to divide into a different kernel :(
    //Chunk calculation starts at 1
    for (int i = 1; i <= n_splits; i++) {
        Range curr_range = { i, n_splits };
        int thread_dim = (curr_range.max - curr_range.min) + 1;
        //TODO MAKE MORE SCALABLE THAN USER REQUIREMENT
        if (thread_dim > 1024) {
            printf("Thread dim %d is greater than 1024, increase chunk size!", thread_dim);
            return;
        }
        //Max number of threads per block is 1024t/b
        //printf("%d\n", thread_dim);
        
        //1 Block (maybe increase to increase parallelization)
        printf("------------------Chunk %d started--------------------\n", i);
        ZTestKernel <<<1,thread_dim>>> (i, 
            &thread_dim, 
            chunksize, 
            control_mat, 
            case_mat, 
            zpthres, 
            &sd_tot, 
            d_flag);
        while (*d_flag != 1) {}
        *d_flag = 0;
        printf("------------------Chunk %d finished--------------------\n", i);
    }
    

    printf("-----------KERNEL FINISHED-------------\n");
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE************
cudaError_t EpiScan(const Matrix genotype_data, 
                    const Matrix phenotype_data, 
                    const double zpthres, 
                    const int chunksize) {
    double* d_zpthres;
    int* d_geno_height;
    int* d_geno_width;
    int* d_pheno_height;
    int* d_pheno_width;
    int* d_chunksize;
    int* d_flag;

    cudaError_t cudaStatus;

    printf("EpiScan called!\n");

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    printf("Start data allocation/movement\n");

    //Setting size of the heap
    size_t heapSizeInBytes = 3000000000; //This is a guestimate

    // Set the heap size limit
    cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSizeInBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSetLimit failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    //Determine the case and control matrices
    int control_count = 0, case_count = 0;
    for (int i = 0; i < phenotype_data.height; i++) {
        if (phenotype_data.elements[i] == 0.0) control_count++;
        else case_count++;
    }

    Matrix control_mat{ genotype_data.width, control_count, new double[control_count * genotype_data.width] };
    Matrix case_mat{ genotype_data.width, case_count, new double[case_count * genotype_data.width] };

    // M(row, col) = *(M.elements + row * M.width + col)
    control_count = (case_count = 0);
    for (int i = 0; i < phenotype_data.height; i++) {
        if (phenotype_data.elements[i] == 0.0) {
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

    // Load case to device memory
    Matrix d_case = {case_mat.width, case_mat.height, new double[case_mat.width * case_mat.height]};
    size_t size = case_mat.width * case_mat.height * sizeof(double); //Calculate the total amount of memory to allocate for matrix A
    cudaStatus = cudaMalloc(&d_case.elements, size); //Allocate the data on the CUDA device
    if (cudaStatus != cudaSuccess) {
        printf("d_case cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_case.elements, case_mat.elements, size,
        cudaMemcpyHostToDevice); //Copy the memory stored in the Matrix struct into the allocated memory
    if (cudaStatus != cudaSuccess) {
        printf("d_case cudaMemcpy failed!");
        goto Error;
    }

    // Load control to device memory
    Matrix d_control = { control_mat.width, control_mat.height, new double[control_mat.width * control_mat.height]};
    size = control_mat.width * control_mat.height * sizeof(double); //Calculate the total amount of memory to allocate for matrix A
    cudaStatus = cudaMalloc(&d_control.elements, size); //Allocate the data on the CUDA device
    if (cudaStatus != cudaSuccess) {
        printf("d_control cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_control.elements, control_mat.elements, size,
        cudaMemcpyHostToDevice); //Copy the memory stored in the Matrix struct into the allocated memory
    if (cudaStatus != cudaSuccess) {
        printf("d_control cudaMemcpy failed!");
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

    //--------------------------------------------------------------------------------
    //Load d_geno_height to memory device
    cudaStatus = cudaMalloc((void**)&d_geno_height, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("d_geno_height cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_geno_height, &genotype_data.height, sizeof(int),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("d_geno_height cudaMemcpy failed!");
        goto Error;
    }

    //Load d_geno_width to memory device
    cudaStatus = cudaMalloc((void**)&d_geno_width, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("d_geno_width cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_geno_width, &genotype_data.width, sizeof(int),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("d_geno_width cudaMemcpy failed!");
        goto Error;
    }

    //Copy the value of geno_width to __constant__ memory
    cudaMemcpyToSymbol(n_SNP, &genotype_data.width, sizeof(int)); //Not real error

    //Load d_pheno_height to memory device
    cudaStatus = cudaMalloc((void**)&d_pheno_height, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("d_pheno_height cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_pheno_height, &phenotype_data.height, sizeof(int),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("d_pheno_height cudaMemcpy failed!");
        goto Error;
    }

    //Load d_pheno_width to memory device
    cudaStatus = cudaMalloc((void**)&d_pheno_width, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("d_pheno_width cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_pheno_width, &phenotype_data.width, sizeof(int),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("d_pheno_width cudaMemcpy failed!");
        goto Error;
    }

    //Load d_flag to memory device
    cudaStatus = cudaMalloc((void**)&d_flag, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("d_pheno_width cudaMalloc failed!");
        goto Error;
    }

    printf("Memory Allocated!\n");

    size_t freeMemory, totalMemory;
    cudaStatus = cudaMemGetInfo(&freeMemory, &totalMemory);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed! %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    printf("Total GPU Memory: %zu bytes\n", totalMemory);
    printf("Free GPU Memory: %zu bytes\n", freeMemory);

    // Invoke kernel
    //Only need 1 thread and 1 block because I will be parallelizing the rest within this kernel
    EpiScanKernel <<<1,1>>> (d_case, 
                             d_control, 
                             d_zpthres, 
                             d_chunksize, 
                             d_geno_height, 
                             d_geno_width, 
                             d_pheno_height, 
                             d_pheno_width, 
                             d_flag);
    cudaDeviceSynchronize();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "EpiScanKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaFree(&d_case);
    cudaFree(&d_control);
    cudaFree(d_zpthres);
    cudaFree(d_geno_height);
    cudaFree(d_geno_width);
    cudaFree(d_pheno_height);
    cudaFree(d_pheno_width);
    cudaFree(d_chunksize);
    cudaFree(d_flag);

Error:
    cudaFree(&d_case);
    cudaFree(&d_control);
    cudaFree(d_zpthres);
    cudaFree(d_geno_height);
    cudaFree(d_geno_width);
    cudaFree(d_pheno_height);
    cudaFree(d_pheno_width);
    cudaFree(d_chunksize);
    cudaFree(d_flag);

    return cudaStatus;
}
