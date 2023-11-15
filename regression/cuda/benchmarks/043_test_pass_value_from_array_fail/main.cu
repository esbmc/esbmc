//fail
//--blockDim=64 --gridDim=64 --no-inline

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#define N 2//64

__device__ void bar(float x) {
	assert(0);
}

__global__ void foo(int* A) {

  bar(A[0]);

}

int main(){

	int *b;
	int *dev_b;

	b = (int*)malloc(N*sizeof(int));

	for (int i = 0; i < N; ++i){
		b[i] = i+1;
		printf(" %d; ", b[i]);
	}

	cudaMalloc((void**)&dev_b, N*sizeof(float));

	cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

		//foo<<<1,N>>>(dev_b);
		ESBMC_verify_kernel(foo,1,N,dev_b);	

	free(b);
	cudaFree(dev_b);
}
