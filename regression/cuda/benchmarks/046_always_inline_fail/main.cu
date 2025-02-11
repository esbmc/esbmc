#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime_api.h>

#define N 2//64

__device__ int bar() __attribute__((always_inline));

__device__ int bar()
{
  return 5;
}

__global__ void foo()
{
  int x = bar();
  assert(x != 5);
  //printf("%d ", x);

}

int main(){

	//foo<<<1, N>>>();
	ESBMC_verify_kernel(foo,1,N);

	cudaThreadSynchronize();

	return 0;
}
