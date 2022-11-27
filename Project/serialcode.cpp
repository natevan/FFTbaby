//Jose Soto and Nathan Van De Vyvere

//This program will compute the Monte Carlo estimation of
//4 randomly generated polynomials 
#include <math.h>
#include <random>
#include <stdio.h>
#include <time.h>

using namespace std;

//Number of samples to be computer per polynomial
#define samples 8192
//Bounding points of polynomial
#define a 1
#define b 5
//Number of polynomials to be computed
#define p 5

//Struck that will contain random coefficiants
struct polynomial {
  int three;
  int two;
  int one;
};

int main() {
  //Initialize random engine and polynomial class
  default_random_engine rng;
  uniform_int_distribution<> coefficient(0, 9);
  polynomial exps[p];
  for (int i = 0; i < p; i++) {
    exps[i].three = coefficient(rng);
    exps[i].two = coefficient(rng);
    exps[i].one = coefficient(rng);
  }
  printf("Samples Processed: %d\n\n", samples);
  //Timimg variables
  clock_t start, end;
  double cpu_time_used;
  start = clock();
  //Bounding box variables for Monte Carlo
  double x, y;
  double bounding_box, box_height, box_width, upper_limit, answer[p], local_max;
  int count;
  local_max = box_height = 0;
  uniform_real_distribution<double> x_sample(a, b);
  box_width = b - a;
  //Loop for every polynomial
  for (int k = 0; k < p; k++) {
    answer[k] = 0;
    //Loop for every sample
    for (int i = 0; i < samples; i++) {
      count = 0;
      //Create a bounding box
      for (int j = 0; j < samples; j++) {
        x = x_sample(rng);
        local_max = exps[k].three * pow(x, 5) + exps[k].two * pow(x, 3) +
                    exps[k].one * pow(x, 4);
        box_height = max(local_max, box_height);
      }
      //Area of boudning box
      bounding_box = box_height * box_width;
      uniform_real_distribution<double> y_sample(0, box_height);
      count = 0;
      //Simulate random points
      for (int j = 0; j < samples; j++) {     
        x = x_sample(rng);
        y = y_sample(rng);
        upper_limit = exps[k].three * pow(x, 5) + exps[k].two * pow(x, 3) +
                      exps[k].one * pow(x, 4);
        //Check if random pointed landed under the curve
        if (y <= upper_limit) {
          count++;
        }
      }
      //Summation of Probabilities
      answer[k] += ((double)count / samples) * bounding_box;
    }
  }
  //Get the average answer
  for (int i = 0; i < p; i++) {
    answer[i] /= samples;
  }
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  //Printing routine
  for (int i = 0; i < p; i++) {
    printf("Expression: %dx^5 + %dx^3 + %dx^4\n", exps[i].three, exps[i].two, exps[i].one);
    printf("Result: %f\n\n", answer[i]);
  }
  printf("Time used %f\n", cpu_time_used);
  printf("Average time used %f", cpu_time_used / p);
  return 0;
}