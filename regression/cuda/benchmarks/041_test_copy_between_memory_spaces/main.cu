//--blockDim=64 --gridDim=1 --equality-abstraction --no-inline
#include "cuda_runtime_api.h"
#include <stdio.h>
#include <assert.h>
#define N 2

__global__ void foo(int* p) {

	__shared__  int A[10];

	A[0] = 1;

	p[0] = A[0];

}

int main(){

	int *b;
	int *dev_b;

	b = (int*)malloc(N*sizeof(int));

	for (int i = 0; i < N; ++i){
		b[i] = i+1;
		printf(" %d; ", b[i]);
	}
	printf("\n");

	cudaMalloc((void**)&dev_b, N*sizeof(int));

	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

    // foo<<<1,N>>>(dev_b);
	ESBMC_verify_kernel(foo,1,N,dev_b);

	cudaMemcpy(b, dev_b, N*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i){
		printf(" %d; ", b[i]);
		assert(b[0]==1);
	}

	free(b);
	cudaFree(dev_b);

}
