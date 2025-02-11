#include <stdio.h>
#include <assert.h>
#include "cuda_runtime_api.h"

#define N 2

__global__ void foo(int* p) {
    p[threadIdx.x] = 2;
    __syncthreads();
}

int main(){

	int *a;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(N*size);

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	//foo<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(foo,1,N,dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++){
		assert(a[i]==1);
	}

	free(a);

	cudaFree(dev_a);

	return 0;
}
