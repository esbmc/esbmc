//xfail:data-race
// Write by thread 0
// Write by thread 1
// x = 1

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define N 2//512

__global__ void example(float * A, int x) {

	//__requires(x = 1); // x deve ser a diferença entre o limite do if1 e do if2

    if(threadIdx.x == 0) {
        A[threadIdx.x + x] = threadIdx.x; //A[1] = 0;
    }

    if(threadIdx.x == 1) {
        A[threadIdx.x] = threadIdx.x; //A[1] = 1;
   }
}

int main() {
	int c=1;
	float *a;
	float *dev_a;

	a = (float*)malloc(N*sizeof(float));

	cudaMalloc((void**)&dev_a, N*sizeof(float));

	cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);

	// example<<<1, N>>>(dev_a, c);
	ESBMC_verify_kernel_fuintint(example,1, N,dev_a, c);

	cudaMemcpy(a, dev_a, N*sizeof(float), cudaMemcpyDeviceToHost);

	assert(a[1] == 0 || a[1] == 1);	

	free(a);
	cudaFree(dev_a);

	return 0;
}
