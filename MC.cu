#include <stdio.h>
#include <random>
#include <math.h>
#include <curand_kernel.h>

using namespace std;

#define samples 8192
#define a 2
#define b 4

__global__
void MC(double* answer_d){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(123, i, 654, &state);

    double x, y;
    double bounding_box, box_height, box_width, upper_limit, answer, local_max;
    int count;

    local_max = box_height = 0;
    box_width = b - a;
    // y = x^4
    for (int i = 0; i < 100; i++)
    {
        count = 0;
        for (int j = 0; j < samples; j++)
        {
            x = (b-a)*curand_uniform(&state)+a;
            local_max = pow(x, 4);
            box_height = max(local_max, box_height);
        }
        bounding_box = box_height * box_width;
        
        count = 0;
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
        answer_d[i] = ((double)count / samples) * bounding_box;
    }
}

int main()
{
    int size = sizeof(double)*samples;
    double *answer = (double*)malloc(size);
    double *answer_d;
    cudaMalloc((void**) &answer_d, size);
    dim3 block(1, 1);
    dim3 blockdim(1024, 1);
    MC <<< block, blockdim>>> (answer_d);
    cudaDeviceSynchronize();
    cudaMemcpy(answer, answer_d, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i<samples; i++)
    {
        printf("%f\n", answer[i]);
    }
    return 0;
}
