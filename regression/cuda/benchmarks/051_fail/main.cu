#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cuda_runtime_api.h"

//pass
//--blockDim=10 --gridDim=64 --no-inline

__global__ void foo() {

  __shared__ int A[10][10];
  A[threadIdx.y][threadIdx.x] = 2;
  assert(A[threadIdx.y][threadIdx.x]!=2);
}

int main(){

	dim3 dimBlock(2,2);
	//foo<<<1, dimBlock>>>();
	ESBMC_verify_kernel(foo, 1, dimBlock);
	cudaThreadSynchronize();
}

