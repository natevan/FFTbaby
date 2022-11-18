#include <stdio.h>
#include <random>
#include <math.h>
#include <curand_kernel.h>

using namespace std;

#define samples 1024
#define a 0
#define b 2

__global__
void MC(double* answer_d){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(123, i, 654, &state);

    double x, y;
    double bounding_box, box_height, box_width, upper_limit, local_max;
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
    
    //count = 9;
    for (int j = 0; j < samples; j++)
    {
        x = (b-a)*curand_uniform(&state)+a;
        y = box_height * curand_uniform(&state);
        upper_limit = pow(x, 4);
        if (y <= upper_limit)
        {
            count++;
            //printf("count++ %d\n", count);
        }
    }
    //printf("Bounding Box After%f\n", (double(count)/samples)*bounding_box);
    answer_d[i] = (double(count)/samples)*bounding_box; 
    //printf("%f", answer_d[i]);
    /*
    if(i == 0)
    {
        printf("count %d samples %f box height %f  box width %f bounding box %f\n", count, samples, box_height, box_width, bounding_box);
    }
    answer_d[i] = ((double)count / samples) * bounding_box;
    */
}

int main()
{
    int size = sizeof(double)*samples;
    double *answer = (double*)malloc(size);
    double *answer_d;
    double sum = 0;
    cudaMalloc((void**) &answer_d, size);
    dim3 block(1, 1);
    dim3 blockdim(1024, 1);
    MC <<< block, blockdim>>> (answer_d);
    cudaDeviceSynchronize();
    cudaMemcpy(answer, answer_d, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i<samples; i++)
    {
        sum += answer[i];
    }
    double avg = sum/samples;
    printf("Answer: %f\n", avg);
    return 0;
}
