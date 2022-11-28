/***************************************************************************
**  HPC Project -- Monte Carlo Method
**  Jose Soto and Nathan Van De Vyvere
**  November 28, 2022
****************************************************************************
**  This program will approimate integrals on randomly generated 3 term 
**  polynomials using the Monte Carlo mehtod.
****************************************************************************
**  Runs on maverick2
**  sbatch gpuScript1
***************************************************************************/
#include <stdio.h>
#include <random>
#include <math.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

using namespace std;

// error handling and debugging
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}

// number of expressions
#define PROBLEM_SIZE 5
#define BLOCKS_PER_PROBLEM 8
#define MAX_THREADS 1024
#define MAX_BLOCKS PROBLEM_SIZE*BLOCKS_PER_PROBLEM
#define TOTAL_THREADS MAX_BLOCKS*MAX_THREADS
#define samples BLOCKS_PER_PROBLEM*MAX_THREADS
#define a 1
#define b 5

struct polynomial
{
    int three;
    int two;
    int one;
};

__global__
void MC(double* answer_d, polynomial * exps){
    
    int i = (threadIdx.x + blockIdx.x * blockDim.x);
    // cyclical pattern for blocks assigned to each polynomial
    int offset = blockIdx.x % PROBLEM_SIZE;
    polynomial P = exps[offset];
    // CUDA random number generation, generates sequence based on global index
    curandState state;
    curand_init(123, i, 654, &state);
    double x, y;
    double bounding_box, box_height, box_width, upper_limit, local_max;
    int count = 0;
    local_max = box_height = 0;
    box_width = b - a;
    // monte carlo, random sampling within a determined bounding box
    for (int j = 0; j < samples; j++)
    {
        x = (b-a)*curand_uniform(&state)+a;
        local_max = P.three * pow(x, 3) + P.two * pow(x, 2) + P.one * x;
        box_height = max(local_max, box_height);
    }
    bounding_box = box_height * box_width;
    for (int j = 0; j < samples; j++)
    {
        x = (b-a)*curand_uniform(&state)+a;
        y = box_height * curand_uniform(&state);
        upper_limit = P.three * pow(x, 3) + P.two * pow(x, 2) + P.one * x;
        if (y <= upper_limit)
        {
            count++;
        }
    }
    // takes the average integral to be summed by the host
    answer_d[i] = ((double(count)/double(samples))*bounding_box) / double(samples);
}


int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // var for allocating memory
    int size = sizeof(double)*TOTAL_THREADS;
    int poly_size = sizeof(polynomial)*PROBLEM_SIZE;
    // allocates the array to hold local answers
    double *answer = (double*)malloc(size);
    // random number generation engine
    default_random_engine rng;
    // sets the distribution of random numbers
    uniform_int_distribution<> coefficient(0,9);
    polynomial exps[PROBLEM_SIZE], *exps_d;
    // generates random coefficient values for expressions
    for (int i = 0; i < PROBLEM_SIZE; i++)
    {
        exps[i].three = coefficient(rng);
        exps[i].two = coefficient(rng);
        exps[i].one = coefficient(rng);
    }
    double *answer_d;
    double sum[PROBLEM_SIZE];
    // allocates memory in the GPU
    CUDA_CALL(cudaMalloc((void**) &answer_d, size));
    CUDA_CALL(cudaMalloc((void**) &exps_d, poly_size));
    // copies the values of the generated coefficients to global GPU memory
    CUDA_CALL(cudaMemcpy((void *)exps_d, (void*)exps, poly_size, cudaMemcpyHostToDevice));
    // allocates the number of blocks and threads per block
    dim3 block(MAX_BLOCKS, 1);
    dim3 blockdim(MAX_THREADS, 1);
    printf("Launching kernel with %d blocks with %d threads each\n", MAX_BLOCKS, MAX_THREADS);
    printf("Samples Computed: %d\n", samples);
    cudaEventRecord(start);
    // calls the GPU kernal
    MC <<< block, blockdim>>> (answer_d, exps_d);
    // cudaDeviceSynchronize();
    // copies answer array from GPU back to host memory
    CUDA_CALL(cudaMemcpy(answer, answer_d, size, cudaMemcpyDeviceToHost));
    // deallocates GPU memory
    cudaFree(answer_d);
    cudaFree(exps_d);
    // sums the averages of the estimated integrals for each expression
    int offset;
    for(int i = 0; i < MAX_BLOCKS; i++)
    {
        offset = MAX_THREADS * i;
        for(int j = 0; j < MAX_THREADS; j++)
        {
            sum[i % PROBLEM_SIZE] += answer[j + offset];
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    for (int i = 0; i < PROBLEM_SIZE; i++)
    {
        printf("Coefficients: %d %d %d  \n", exps[i].three, exps[i].two, exps[i].one);
        printf("Answer %d: %f\n", i+1, sum[i]);
    }
    printf("Time taken: %f ms", time);
    free(answer);
    return 0;
}
