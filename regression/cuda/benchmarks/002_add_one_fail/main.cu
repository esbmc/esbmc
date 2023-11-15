//fail: array out of bounds

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define N 2

__global__ void race_test (unsigned int* i, int* A)
{
  int tid = threadIdx.x;
  int j = atomicAdd(i,1);
  A[j] = tid;
}

int main(){

	unsigned int *i;
	int *A;
	unsigned int *dev_i;
	int *dev_A;

	A = (int*)malloc(N*sizeof(int));
	i = (unsigned int*)malloc(sizeof(unsigned int));

	for (int t=0; t<N; t++)
		*(A+t) = 4;

	cudaMalloc((void**)&dev_A, N*sizeof(int));
	cudaMalloc((void**)&dev_i, sizeof(unsigned int));

	cudaMemcpy(dev_A, A, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_i, i, sizeof(unsigned int), cudaMemcpyHostToDevice);

	// race_test<<<N,1>>>(dev_i, dev_A);
	ESBMC_verify_kernel_u(race_test,1,N,dev_i,dev_A);

	cudaMemcpy(A, dev_A, N*sizeof(int), cudaMemcpyDeviceToHost);

	for (int t=0; t<N; t++)
		assert(A[t]==0 || A[t]==1);

	free(A);
	free(i);
	cudaFree(dev_A);
	cudaFree(dev_i);

	return 0;
}
