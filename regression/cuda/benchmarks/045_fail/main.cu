#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define N 2//64

__device__ int bar () {
  return 0;
}

__global__ void foo() {
  assert(bar () !=0);
}

int main(){

	//foo<<<1, N>>>();
	ESBMC_verify_kernel(foo, 1, N);

	cudaThreadSynchronize();

	return 0;
}
