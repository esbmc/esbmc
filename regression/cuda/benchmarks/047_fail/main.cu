#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime_api.h"
#include <assert.h>

#define N 2//64

__global__ void foo(int *c) {
  int b, a;
  a = 2;
  b = 3;
  c[threadIdx.x]= a+b;
  __syncthreads ();
}

int main(){
	int *a;
	int *dev_a;

	a = (int*)malloc(N*sizeof(int));

	cudaMalloc((void**)&dev_a, N*sizeof(int));

	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);

	//foo<<<1, N>>>(dev_a);
	ESBMC_verify_kernel(foo,1,N,dev_a);

	cudaMemcpy(a, dev_a, N*sizeof(int), cudaMemcpyDeviceToHost);

	for (int t=0;t<N;t++){
		//printf ("%d ", a[t]);
		assert(a[t]!=5);
	}

	cudaFree(dev_a);
	free(a);
	return 0;
}
