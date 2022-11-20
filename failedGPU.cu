#include <stdio.h>
#include <random>
#include <math.h>
#include <curand_kernel.h>

using namespace std;

#define SAMPLES 10000
#define PROBLEM_SIZE 8192
#define MAX_THREADS 1024
#define a 0
#define b 2

struct polynomial
{
    int five;
    int four;
    int three;
    int two;
    int one;
};

__global__
void MC(double * answer_d, polynomial * exps){
    // globa index
    int i = (threadIdx.x + blockIdx.x * blockDim.x);
    // state for random number generation
    curandState state;
    // initalizes sequence for random values
    curand_init(123, i, 654, &state);
    // gets expression for specific block
    polynomial P = exps[blockIdx.x];
    // shared value for block
    __shared__ int block_answer;
    double x, y;
    double bounding_box, box_height, box_width, upper_limit, local_max, local_answer;
    int count = 0;
    local_max = box_height = 0;
    box_width = b - a;
    
    for (int j = 0; j < SAMPLES; j++)
    {
        x = (b-a)*curand_uniform(&state)+a;
        local_max = P.five * pow(x, 5) + P.four * pow(x, 4) + P.three * pow(x, 3) + P.two * pow(x, 2) + P.one * x;
        box_height = max(local_max, box_height);
    }
    bounding_box = box_height * box_width;
    for (int j = 0; j < SAMPLES; j++)
    {
        x = (b-a)*curand_uniform(&state)+a;
        y = box_height * curand_uniform(&state);
        upper_limit = P.five * pow(x, 5) + P.four * pow(x, 4) + P.three * pow(x, 3) + P.two * pow(x, 2) + P.one * x;
        if (y <= upper_limit)
        {
            printf("count= %d thread= %d",count,i);
            count++;
        }
    }
    local_answer = (double(count)/SAMPLES)*bounding_box;
    block_answer += local_answer;
    __syncthreads();
    if(!threadIdx.x)
    {
        answer_d[blockIdx.x] = block_answer;
    }
}

int main()
{
    default_random_engine rng;
    uniform_int_distribution<> coefficient(0,9);
    polynomial exps[PROBLEM_SIZE], *exps_d;
    // generates random coefficient values for expressions
    for (int i = 0; i < PROBLEM_SIZE; i++)
    {
        exps[i].five = coefficient(rng);
        exps[i].four = coefficient(rng);
        exps[i].three = coefficient(rng);
        exps[i].two = coefficient(rng);
        exps[i].one = coefficient(rng);
    }

    int size_double = sizeof(double) * PROBLEM_SIZE;
    int size_poly = sizeof(polynomial) * PROBLEM_SIZE;
    double *answer = (double*)malloc(size_double);
    double *answer_d;
    cudaMalloc((void**) &answer_d, size_double);
    cudaMalloc((void**) &exps_d, size_poly);
    cudaMemcpy(exps_d, exps, size_poly, cudaMemcpyHostToDevice);
    dim3 MC_GRID(PROBLEM_SIZE, 1);
    dim3 MC_BLOCK(MAX_THREADS, 1);
    
    printf("Launching kernel with %d blocks with %d threads each\n", PROBLEM_SIZE, MAX_THREADS);
    MC <<< MC_GRID, MC_BLOCK >>> (answer_d, exps_d);
    cudaDeviceSynchronize();
    printf("SAMPLES Computed: %d\n", SAMPLES);
    

    cudaMemcpy(answer, answer_d, size_double, cudaMemcpyDeviceToHost);
    cudaFree(answer_d);
    cudaFree(exps_d);
    for (int i = 0; i < PROBLEM_SIZE; i++)
    {
        printf("Coefficients: %d %d %d %d %d -- ", exps[i].five, exps[i].four, exps[i].three, exps[i].two, exps[i].one);
        printf("Answer: %f\n", answer[i]);
    }
    free(answer);
    return 0;
}
