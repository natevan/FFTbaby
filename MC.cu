#include <stdio.h>
#include <random>
#include <math.h>
#include <curand_kernel.h>

using namespace std;

#define samples 131072
#define MAX_BLOCKS 64
#define MAX_THREADS 1024
#define a 0
#define b 2

__global__
void MC(double* answer_d, int multiple){
    curandState state;
    int i = (threadIdx.x + blockIdx.x * blockDim.x);
    curand_init(123, i, 654, &state);
    int offset = MAX_THREADS * MAX_BLOCKS;

    double x, y;
    double bounding_box, box_height, box_width, upper_limit, local_max;
    for(int m = 0; m < multiple; m++)
    {
        int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (offset * m);
        int count = 0;
        local_max = box_height = 0;
        box_width = b - a;
        // y = x^4
        for (int j = 0; j < samples; j++)
        {
            x = (b-a)*curand_uniform(&state)+a;
            local_max = pow(x, 4);
            box_height = max(local_max, box_height);
        }
        bounding_box = box_height * box_width;
        for (int j = 0; j < samples; j++)
        {
            x = (b-a)*curand_uniform(&state)+a;
            y = box_height * curand_uniform(&state);
            upper_limit = pow(x, 4);
            if (y <= upper_limit)
            {
                count++;
            }
        }
        answer_d[idx] = (double(count)/samples)*bounding_box; 
    }
}

int main()
{
    int size = sizeof(double)*samples;
    double *answer = (double*)malloc(size);
    double *answer_d;
    double sum = 0;
    cudaMalloc((void**) &answer_d, size);
    dim3 block(MAX_BLOCKS, 1);
    dim3 blockdim(MAX_THREADS, 1);
    int MULTIPLE = samples/(MAX_BLOCKS*MAX_THREADS);
    printf("Luanching kernel with %d blocks with %d threads each\n", MAX_BLOCKS, MAX_THREADS);
    printf("Samples Computed: %d\n", samples);
    MC <<< block, blockdim>>> (answer_d, MULTIPLE);
    cudaDeviceSynchronize();
    cudaMemcpy(answer, answer_d, size, cudaMemcpyDeviceToHost);
    cudaFree(answer_d);
    for (int i = 0; i<samples; i++)
    {
        sum += answer[i];
    }
    double avg = sum/samples;
    printf("Answer: %f\n", avg);
    free(answer);
    return 0;
}
