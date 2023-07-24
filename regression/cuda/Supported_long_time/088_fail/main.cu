//fail
//--blockDim=1024 --gridDim=1024 --no-inline

#include <stdio.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime_api.h>
#include <math_functions.h>

#define DIM 2 //1024 in the future
#define N 2//DIM*DIM

__global__ void mul24_test (int* A)
{
  int idxa = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

  A[idxa] = idxa;
}

int main (){
	int *a;
	int *dev_a; 
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 1;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	//mul24_test<<<DIM,DIM>>>(dev_a,dev_b);
	ESBMC_verify_kernel(mul24_test,1,N,dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
		assert (a[i] != i);	


	free(a);

	cudaFree(dev_a);

	return 0;
}
