#include <stdio.h>
#include <random>
#include <math.h>
#include <time.h>

using namespace std;

#define samples 8192
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

int main()
{
    default_random_engine rng;
    uniform_int_distribution<> coefficient(0,9);
    polynomial exps[samples];
    for (int i = 0; i < samples; i++)
    {
        exps[i].five = coefficient(rng);
        exps[i].four = coefficient(rng);
        exps[i].three = coefficient(rng);
        exps[i].two = coefficient(rng);
        exps[i].one = coefficient(rng);
    }
    // printf("%d %d %d %d %d\n",exps[0].five,exps[0].four,exps[0].three,exps[0].two,exps[0].one);
    // printf("%d %d %d %d %d\n",exps[100-1].five,exps[100-1].four,exps[100-1].three,exps[100-1].two,exps[100-1].one);

    clock_t start, end;
    double cpu_time_used;
    start = clock();
    double x, y;
    double bounding_box, box_height, box_width, upper_limit, answer [samples], local_max;
    int count;
    local_max = box_height = 0;
    uniform_real_distribution<double> x_sample(a, b);
    box_width = b - a;
    for (int k = 0; k < samples; k++)   
    {
        answer[k] = 0;
        for (int i = 0; i < samples; i++)
        {
            count = 0;
            for (int j = 0; j < samples; j++)
            {
                x = x_sample(rng);
                local_max = exps[k].five * pow(x, 5) + exps[k].four * pow(x, 4) + exps[k].three * pow(x, 3) + \
                exps[k].two * pow(x, 2) + exps[k].one * x;
                box_height = max(local_max, box_height);
            }
            bounding_box = box_height * box_width;
            uniform_real_distribution<double> y_sample(0, box_height);
            count = 0;
            for (int j = 0; j < samples; j++)
            {
                x = x_sample(rng);
                y = y_sample(rng);
                upper_limit = exps[k].five * pow(x, 5) + exps[k].four * pow(x, 4) + exps[k].three * pow(x, 3) + \
                exps[k].two * pow(x, 2) + exps[k].one * x;
                if (y <= upper_limit)
                {
                    count++;
                }
            }
            answer[k] += ((double)count / samples) * bounding_box;
        }
    }

    for(int i = 0; i < samples; i++)
    {
        answer[i] /= samples;
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("%f\n", answer[0]);
    printf("%f\n", answer[samples-1]);
    printf("Time used %f", cpu_time_used);
    return 0;
}

