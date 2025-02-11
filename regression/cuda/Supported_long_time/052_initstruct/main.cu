//pass
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define N 2//32

__global__ void kernel(uint4 *out) {
  uint4 vector = {1,1,1,1};
  out[threadIdx.x] = vector;
}

int main(){
	uint4 *a;
	uint4 *dev_a;

	a = (uint4*)malloc(N*sizeof(uint4));

	cudaMalloc((void**)&dev_a, N*sizeof(uint4));

	cudaMemcpy(dev_a, a, N*sizeof(uint4), cudaMemcpyHostToDevice);

		//kernel<<<1, N>>>(dev_a);
		ESBMC_verify_kernel_u(kernel,1,N,dev_a);

	cudaMemcpy(a, dev_a, N*sizeof(uint4), cudaMemcpyDeviceToHost);

	for (int i=0; i<N; i++) {
		assert(a[i].x == 1);			
		assert(a[i].y == 1);
		assert(a[i].z == 1);			
		assert(a[i].w == 1);
	}

	free(a);
	cudaFree(dev_a);

}
