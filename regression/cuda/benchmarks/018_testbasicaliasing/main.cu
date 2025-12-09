//fail: data race
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime_api.h"
#include <assert.h>

#define N 2//64

__global__ void foo (int* p, int* q){

    p[2] = q[2] + 1;

}

int main() {
	int *a;
	int *dev_a;
	int *b;
	int *dev_b;

	a = (int*)malloc(N*sizeof(int));
	b = (int*)malloc(N*sizeof(int));

	for (int i=0; i<N; i++){
		a[i]=i;
		b[i]=2*i;
	}

	cudaMalloc((void**)&dev_a, N*sizeof(int));
	cudaMalloc((void**)&dev_b, N*sizeof(int));

	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	//foo<<<N, N>>>(dev_a, dev_b);
	ESBMC_verify_kernel(foo, 1, N, dev_a, dev_b);

	cudaMemcpy(a, dev_a, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, dev_b, N*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i=0; i<N; i++){
		printf ("a[%d]= %d; b[%d]=%d;\n", i, a[i], i, b[i]);
	}

	assert(a[2]==(b[2]+1));

	free(a); free(b);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return 0;
}
