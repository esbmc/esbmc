//pass
//--blockDim=64 --gridDim=64 --no-inline

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <assert.h>

#define N 2//64

__device__ int f(int x) {

  return x + 1;
}

__global__ void foo(int *y) {

	*y = f(2);

}

int main() {
	int *a = (int*)malloc(sizeof(int));
	int *dev_a;

	cudaMalloc((void**)&dev_a, sizeof(int));
	
	//foo<<<1, N>>>(dev_a);
		ESBMC_verify_kernel(foo, 1, N, dev_a);

	cudaMemcpy(a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);

//	printf("%d", *a);

	assert(*a==3);

	free(a);
	cudaFree(dev_a);

	return 0;
}
