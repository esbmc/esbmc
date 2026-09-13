
// No-race companion to 113_curand_race_fail: each thread draws from its own
// generator state.

#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 2 //4

__global__ void curand_test(curandState *state, float *A) {
   A[threadIdx.x] = curand_uniform(&state[threadIdx.x]);
}

int main(){
	float *dev_a;
	curandState *dev_state;

	cudaMalloc((void**) &dev_a, N*sizeof(float));
	cudaMalloc((void**) &dev_state, N*sizeof(curandState));

	//curand_test<<<1,N>>>(dev_state, dev_a);
	ESBMC_verify_kernel(curand_test, 1, N, dev_state, dev_a);

	cudaFree(dev_a);
	cudaFree(dev_state);

	return 0;
}
