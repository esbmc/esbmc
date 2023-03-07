#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void kernel(void) {
  printf("hello world gpu \n");
}
int main() {
  kernel<<<1, 1>>>();
    cudaError_t cudaStatus;

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

  return 0;
}
