//pass
//--blockDim=512 --gridDim=1 --no-inline

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 2 //512

__global__ void curand_test(curandState *state, float *A) {
   A[threadIdx.x] =  curand(&state[threadIdx.x]); // the pseudo random number returned by 'curand' is an unsigned int
}

int main() {
	float *a;
	float *dev_a;
	curandState *dev_state; // is not necessary to initialize dev_state because it is a receptor in the function.

	int size = N*sizeof(float);

	a = (float*)malloc(size);
	cudaMalloc ((void**) &dev_a, size);

	printf("old a:  ");
	for (int i = 0; i < N; i++)
		printf("%f	", a[i]);

	cudaMalloc ( (void**) &dev_state, N*sizeof( curandState ) );

	// curand_test<<<1,N>>>(dev_state, dev_a);
    // ESBMC_verify_kernel(curand_test,1,N,dev_state,dev_a);
	
	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nnew a:  ");
	for (int i = 0; i < N; i++) {
		printf("%f	", a[i]);
		//assert((a[i] == 0 || a[i] == 1)); // we can't put assert() here because we get random numbers
		// maybe we can check if they are > 0 or not NULL... ?
	}
	
	free(a);
	cudaFree(dev_a);
	cudaFree(dev_state);

	return 0;
}
