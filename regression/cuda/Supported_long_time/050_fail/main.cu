//fail
//--blockDim=2048 --gridDim=2 --no-inline
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime_api.h>

#define N 2//2048

__constant__ int A[4096];
__constant__ int B[3] = {0,1,2};

__global__ void kernel(int* x) {
  x[threadIdx.x] = A[threadIdx.x] + B[0]; //permanece constante por ser muito grande. N < 1024 não permanece
}
int main () {

	int *a;
	int *c;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);	

	a = (int*)malloc(size);
	c = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = rand() %10+1;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);	

	//kernel<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(kernel,1,N,dev_a);

	cudaMemcpy(c,dev_a,size,cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++){
		assert(c[i]!=0);
	}
	free(a);
	free(c);
	cudaFree(dev_a);

	return 0;
}
