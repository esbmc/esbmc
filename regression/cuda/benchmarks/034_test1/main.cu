//pass
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define N 2//64

__global__ void foo(int* p) {

  p[threadIdx.x] = 0;

}

int main() {
	int *c;
	int *dev_c;
	c = (int*)malloc(N*sizeof(int));

	for (int i = 0; i < N; ++i)
		c[i] = rand() %10+1;

	cudaMalloc((void**)&dev_c, N*sizeof(int));

	cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);

		//foo<<<1, N>>>(dev_c);
		ESBMC_verify_kernel(foo,1,N,dev_c);

	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i)
			assert(c[i] == 0);
	
	free(c);
	cudaFree(dev_c);

	   return 0;
}
