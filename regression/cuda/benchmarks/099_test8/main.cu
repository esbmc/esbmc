// data-racer
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define N 2

__global__ void foo(int	*p, int *ptr_a) {

	ptr_a = p + threadIdx.x;

}

int main() {
	int *c;
	int *dev_c;
	int *a;
	int *dev_a;


	c = (int*)malloc(N*sizeof(int));
	a = (int*)malloc(N*sizeof(int));

	for (int i = 0; i < N; ++i)
		c[i] = 2;

	cudaMalloc((void**)&dev_c, N*sizeof(int));
	cudaMalloc((void**)&dev_a, N*sizeof(int));

	cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);

    //foo<<<1, N>>>(dev_c, dev_a);
	ESBMC_verify_kernel(foo,1,N,dev_c, dev_a);

	cudaMemcpy(a, dev_a, N*sizeof(int), cudaMemcpyDeviceToHost);

	free(a);
	free(c);
	cudaFree(dev_a);
	cudaFree(dev_c);

	return 0;
}
