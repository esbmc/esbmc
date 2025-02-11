//fail: assertion
//--blockDim=512 --gridDim=1 --no-inline

#include <cuda_runtime_api.h>
#include <assert.h>
#include <stdio.h>

#define N 2//512

__global__ void helloCUDA(volatile int* p)
{
    //__assert(__no_read(p));
    p[threadIdx.x] = threadIdx.x;
}

int main () {
	 int *a;
	 int *dev_a;

	int size = N*sizeof(int);

	a = (int*)malloc(size);
	cudaMalloc ((void**) &dev_a, size);

	//helloCUDA<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(helloCUDA,1,N,dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		assert(!(a[i] == i)); //
	}

	free(a);
	cudaFree(dev_a);

	return 0;
}
