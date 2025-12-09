//pass
//--warp-sync=32 --blockDim=32 --gridDim=1 --equality-abstraction --no-inline
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime_api.h>

#define N 2//32

__global__ void foo(int * A, int * B) {
    A[threadIdx.x] = 1;
    volatile int x = A[threadIdx.x];
    B[threadIdx.x] = 1;
    volatile int y = A[threadIdx.x];
    assert(x==y);
}

int main(){
	int *a, *b;
	int *dev_a, *dev_b;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);

	a = (int*)malloc(size);
	b = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 0;

	for (int i = 0; i < N; i++)
		b[i] = 2;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,size, cudaMemcpyHostToDevice);

	//foo<<<1,N>>>(dev_a, dev_b);
		ESBMC_verify_kernel(foo, 1, N, dev_a, dev_b);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(b,dev_b,size,cudaMemcpyDeviceToHost);

	free(a); free(b);

	cudaFree(dev_a); cudaFree(dev_b);

	return 0;
}
