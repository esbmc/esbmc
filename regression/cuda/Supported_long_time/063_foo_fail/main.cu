#include <stdio.h>
#include "cuda_runtime_api.h"
#include <assert.h>
#define N 2 //16

__device__ int bar(int x) {

	return x + 1;
}

__global__ void foo(int *A) {

	A[threadIdx.x] = bar(threadIdx.x);
}


int main () {

	int *a;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 1;

//	foo<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(foo, 1, N, dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++){
		assert(!(a[i]== (i+1)));
	}

	free(a);
	cudaFree(dev_a);

	return 0;
}
