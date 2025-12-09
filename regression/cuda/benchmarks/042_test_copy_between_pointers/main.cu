//--blockDim=64 --gridDim=64 --equality-abstraction --no-inline

#include "cuda_runtime_api.h"
#include <stdio.h>
#include <assert.h>

#define N 2

__global__ void foo(int* p) {

  __shared__ int A[10];

  int* x;

  x = p;

  	assert(*p <2);

  x[0] = 0;
	
  x = A;

  x[0] = 0;

}

int main(){
	int *b;
	int *dev_b;

	b = (int*)malloc(N*sizeof(int));

	for (int i = 0; i < N; ++i){
		b[i] = i+1;
		printf("%d; ", b[i]);
	}

	printf("\n");

	cudaMalloc((void**)&dev_b, N*sizeof(int));

	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	//foo<<<1,N>>>(dev_b);
	ESBMC_verify_kernel(foo,1,N,dev_b);

	cudaMemcpy(b, dev_b, N*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i){
		printf("%d; ", b[i]);
	}

	assert(b[0]==0);

	free(b);
	cudaFree(dev_b);
}
