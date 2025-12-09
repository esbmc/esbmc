#include <stdio.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <assert.h>

#define N 2//64

__device__ int f(int x) {
	
	return x + 2;
}

__global__ void foo(int *y, int x) {

	*y = f(x);

}

int main() {
	int a=2;
	int b=0;
	int *dev_a;

	cudaMalloc((void**)&dev_a, sizeof(int));

	cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);

	//foo<<<1, N>>>(dev_a, a);
		ESBMC_verify_kernel_intt(foo, 1, N, dev_a, a);

	cudaMemcpy(&b, dev_a, sizeof(int), cudaMemcpyDeviceToHost);

	assert (b == a+2); 

	cudaFree(dev_a);

	return 0;
}
