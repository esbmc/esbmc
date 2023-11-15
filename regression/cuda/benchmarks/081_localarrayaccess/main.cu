//pass 
//--blockDim=10 --gridDim=64 --no-inline

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define N 2

__global__ void foo() {

  __shared__ int A[10];

  A[threadIdx.x] = 2;

  __syncthreads (); //evita corrida de dados

  int x = A[threadIdx.x + 1];

}

int main(){

	//foo<<<1,N>>>();
	ESBMC_verify_kernel(foo,1,N);

	cudaThreadSynchronize();

}
