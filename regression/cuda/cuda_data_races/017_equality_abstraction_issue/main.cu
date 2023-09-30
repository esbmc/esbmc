//xfail:BOOGIE_ERROR
//--warp-sync=32 --blockDim=32 --gridDim=1 --equality-abstraction --no-inline
//kernel.cu:10

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#define N 2//32

__global__ void foo(int * A) {
    A[0] = 1;
    A[1] = 1;
    A[threadIdx.x] = 0;
//__assert(A[0] == 1 | A[1] == 1 | A[2] == 1);
}

int main(){

	int *b;
	int *dev_b;

	b = (int*)malloc(N*sizeof(int));

	for (int i = 0; i < N; ++i){
		b[i] = 2;
	}

	cudaMalloc((void**)&dev_b, N*sizeof(int));
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	//foo<<<1,N>>>(dev_b);
	ESBMC_verify_kernel(foo, 1, N, dev_b);

	cudaMemcpy(b, dev_b, N*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i){
		assert(b[i] == 0 || b[i] == 1);
	}

	free(b);
	cudaFree(dev_b);
}
