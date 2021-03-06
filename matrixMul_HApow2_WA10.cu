// scp /Users/jaehyeon/Documents/CUDA/special_matrix_mul/matrixMul_WA10.cu jaelee@odyssey.rc.fas.harvard.edu:~/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// From mmul_1.cu
// #include <iostream>
// #include <cstdlib>
// #include <ctime>
#include <cublas_v2.h>
// #include <curand.h>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int);


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A 1024 = 32x32
//! @param wA 		  widht of matrix A 16
//! @param hB         height of matrix B 16
//! @param wB         width of matrix B 500
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}
////////////////////////////////////////////////////////////////////////////////

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(cublasHandle_t const &handle, float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

}

///////////////////////////////////////////////////////////////////////////
__global__ void
matrixMul_a16x10_b10xnthreads(float* C, float* A_T, float* B, int WA_T, int WB, int nthreads)
{
    // Note that nthreads has to be a power of 2 and between 16 and 512.

    // Number of registers. Height of submatrix of A or width of submatrix of A_T.
    int nregisters = 16;

    // Block index
    int bx = blockIdx.x; // Index of B matrix sub-block
    int by = blockIdx.y; // Index of A_T matrix sub-block 

    // Thread index
    int tx = threadIdx.x;

    // Register array for the thread. Lenght equal to nregisters.
    float cv[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A, or A_T in fact.
    __shared__ float A_Ts[160]; // nregisters*10

    // First 16 threads reads 10 floats.
    // i < 10 from the width WA.
    for(int i = 0; i < 10; i++){
        if (tx<16){
                A_Ts[tx+i*nregisters] = A_T[i*WA_T+nregisters*by+tx];
            }
        }
    // Loading shared memory
    __syncthreads();

    float *ap = &A_Ts[0];
    float *bp = &B[nthreads*bx+tx]; // memory address of the 
    
    // For each 
    for(int i = 0; i < 10; i++){
        float bv = bp[0];
        cv[0] +=  ap[0] * bv;
        cv[1] +=  ap[1] * bv;
        cv[2] +=  ap[2] * bv;
        cv[3] +=  ap[3] * bv;
        cv[4] +=  ap[4] * bv;
        cv[5] +=  ap[5] * bv;
        cv[6] +=  ap[6] * bv;
        cv[7] +=  ap[7] * bv;
        cv[8] +=  ap[8] * bv;
        cv[9] +=  ap[9] * bv;
        cv[10] +=  ap[10] * bv;
        cv[11] +=  ap[11] * bv;
        cv[12] +=  ap[12] * bv;
        cv[13] +=  ap[13] * bv;
        cv[14] +=  ap[14] * bv;
        cv[15] +=  ap[15] * bv;          
        ap += nregisters;
        bp += WB;
      }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    for(int i=0; i<nregisters; i++){
      C[by*WB*nregisters+bx*nthreads+tx+i*WB] = cv[i];
    }
}



///////////////////////////////////////////////////////////////////////////
__global__ void
matrixMul_a32x10_b10xnthreads(float* C, float* A_T, float* B, int WA_T, int WB, int nthreads)
{
    // Note that nthreads has to be a power of 2 and between 32 and 512.

    // Number of registers. Height of submatrix of A or width of submatrix of A_T.
    int nregisters = 32;

    // Block index
    int bx = blockIdx.x; // Index of B matrix sub-block
    int by = blockIdx.y; // Index of A_T matrix sub-block 

    // Thread index
    int tx = threadIdx.x;

    // Register array for the thread. Length equal to nregisters.
    float cv[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A, or A_T in fact.
    __shared__ float A_Ts[320]; // nregisters*10

    // First 32 threads reads 10 floats.
    // i < 10 from the width WA.
    for(int i = 0; i < 10; i++){
        if (tx<32){
                A_Ts[tx+i*nregisters] = A_T[i*WA_T+nregisters*by+tx];
            }
        }
    // Loading shared memory
    __syncthreads();

    float *ap = &A_Ts[0];
    float *bp = &B[nthreads*bx+tx]; // memory address of the 
    
    // For each 
    for(int i = 0; i < 10; i++){
        float bv = bp[0];
        cv[0] +=  ap[0] * bv;
        cv[1] +=  ap[1] * bv;
        cv[2] +=  ap[2] * bv;
        cv[3] +=  ap[3] * bv;
        cv[4] +=  ap[4] * bv;
        cv[5] +=  ap[5] * bv;
        cv[6] +=  ap[6] * bv;
        cv[7] +=  ap[7] * bv;
        cv[8] +=  ap[8] * bv;
        cv[9] +=  ap[9] * bv;
        cv[10] +=  ap[10] * bv;
        cv[11] +=  ap[11] * bv;
        cv[12] +=  ap[12] * bv;
        cv[13] +=  ap[13] * bv;
        cv[14] +=  ap[14] * bv;
        cv[15] +=  ap[15] * bv;
        cv[16] +=  ap[16] * bv;
        cv[17] +=  ap[17] * bv;
        cv[18] +=  ap[18] * bv;
        cv[19] +=  ap[19] * bv;
        cv[20] +=  ap[20] * bv;
        cv[21] +=  ap[21] * bv;
        cv[22] +=  ap[22] * bv;
        cv[23] +=  ap[23] * bv;
        cv[24] +=  ap[24] * bv;
        cv[25] +=  ap[25] * bv;
        cv[26] +=  ap[26] * bv;
        cv[27] +=  ap[27] * bv;
        cv[28] +=  ap[28] * bv;
        cv[29] +=  ap[29] * bv;
        cv[30] +=  ap[30] * bv;
        cv[31] +=  ap[31] * bv;            
        ap += nregisters;
        bp += WB;
      }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    for(int i=0; i<nregisters; i++){
      C[by*WB*nregisters+bx*nthreads+tx+i*WB] = cv[i];
    }
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;
    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // utilities
    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;

    // set seed for rand()
    srand(2006);

    // Size of arrays
    int WA = 10; 
    int HA = 1024;
    int WB = 512;
    int HB = WA;
    int HC = HA;
    int WC = WB;


    // allocate host memory for matrices A and B and A_T
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    float* h_A_T = (float*) malloc(mem_size_A);    

    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    float flop = 2 * (float)WC * (float)HC * (float)WA;

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // Tranpose the h_A matrix and save in h_A_T
    // Row element i h_A_T:
    for(int i=0; i<WA; i++){
        // Column element j of h_A_T
        for(int j=0; j<HA; j++){
            h_A_T[i*HA+j] = h_A[j*WA+i];
        }
    }

    // // print and compare the two matrices
    // printf("Matrix h_A");
    // for(int i=0; i<HA; i++){
    //     for(int j=0; j<WA; j++){
    //         printf("%.3f ", h_A[i*WA+j]);
    //     }
    //     printf("\n");
    // }
    // // print and compare the two matrices
    // printf("Matrix h_A_T");
    // for(int i=0; i<WA; i++){
    //     for(int j=0; j<HA; j++){
    //         printf("%.3f ", h_A_T[i*HA+j]);
    //     }
    //     printf("\n");
    // }

    // allocate device memory
    float* d_A_T;
    cudaMalloc((void**) &d_A_T, mem_size_A);
    float* d_B;
    cudaMalloc((void**) &d_B, mem_size_B);

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);

    // compute reference solution
    float* reference = (float*) malloc(mem_size_C);    
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);     
    computeGold(reference, h_A, h_B, HA, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Naive CPU (Golden Reference)\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);

   //****************************************************/
    //*  CUBLAS      */
    //****************************************************/
    cudaMemcpy(d_A_T, h_A_T, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);        
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);     
    // Using Cublas
    gpu_blas_mmul(handle, d_A_T, d_B, d_C, HA, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);    
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);    
    // Destroy the handle
    cublasDestroy(handle);    
    printf("Using CUBLAS\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
    // check result
    // printDiff(reference, h_C, WC, HC);    



    dim3 threads,grid;
    int nregisters = 16;
    int nthreads = 8;

    // copy host memory to device
    cudaMemcpy(d_A_T, h_A_T, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);    

    printf("\n");
    printf("# registers = 16 \n");
    //****************************************************/
    //*   a16x10 * b10xnthreads    */
    //****************************************************/
    for(int k=4; k<=9; k++){
        nthreads *= 2; 

        // setup execution parameters
        threads = dim3(nthreads);
        grid = dim3(WB/nthreads, HA/nregisters);

        // create and start timer
        cudaEventCreate(&start);
        cudaEventRecord(start, NULL);     
        // Optimized
        matrixMul_a16x10_b10xnthreads<<< grid, threads >>>(d_C, d_A_T, d_B, HA, WB, nthreads);
        // stop and destroy timer
        cudaEventCreate(&stop);
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msecTotal, start, stop);    
        // copy result from device to host
        cudaMemcpy(h_C, d_C, mem_size_C,
                                  cudaMemcpyDeviceToHost);    
        printf("nthreads %d, nregisters %d on GPU\n", nthreads, nregisters);
        printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
        // check result
        printDiff(reference, h_C, WC, HC);
    
    }



    //****************************************************/
    //*   a32x10 * b10xnthreads    */
    //****************************************************/
    printf("\n");    
    printf("# registers = 32 \n");
    nregisters = 32;
    nthreads = 16; // Min num of threads 32
    for(int k=5; k<=9; k++){
        nthreads *= 2; 
        // setup execution parameters
        threads = dim3(nthreads);
        grid = dim3(WB/nthreads, HA/nregisters);

        // create and start timer
        cudaEventCreate(&start);
        cudaEventRecord(start, NULL);     
        // Optimized
        matrixMul_a32x10_b10xnthreads<<< grid, threads >>>(d_C, d_A_T, d_B, HA, WB, nthreads);
        // stop and destroy timer
        cudaEventCreate(&stop);
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msecTotal, start, stop);    
        // copy result from device to host
        cudaMemcpy(h_C, d_C, mem_size_C,
                                  cudaMemcpyDeviceToHost);    
        printf("nthreads %d, nregisters %d on GPU\n", nthreads, nregisters);
        printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
        // check result
        printDiff(reference, h_C, WC, HC);
    }    

    


    /****************************************************/
    /*  Cleaning                                        */
    /****************************************************/

    // clean up memory
    free(h_A);
    free(h_A_T);
    free(h_B);
    free(h_C);
    free(reference);
    cudaFree(d_A_T);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaThreadExit();


    exit(EXIT_SUCCESS);

}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      if (fabs(data1[k] - data2[k]) > 0.1 ) {
         // printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f \n", i,j, data1[k], data2[k]);
         error_count++;
      }
    }
  }
  printf("Total Errors = %d \n", error_count);
}

