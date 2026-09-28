
//xfail:BOOGIE_ERROR
//--blockDim=2 --gridDim=1 --no-inline
//Write by thread .+kernel.cu:8:4:
// to threadIdx.x != 0 we have 'data race'.

#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 2 //8

__global__ void init_test(curandState *state, unsigned int *A) {
   curand_init(0, 0, 0, state);

   __syncthreads();

   A[threadIdx.x] = curand(&state[threadIdx.x]);
}

int main(){
	unsigned int *dev_a;
	curandState *dev_state;

	cudaMalloc((void**) &dev_a, N*sizeof(unsigned int));
	cudaMalloc((void**) &dev_state, N*sizeof(curandState));

	// init_test<<<1,N>>>(dev_state, dev_a);
	ESBMC_verify_kernel(init_test, 1, N, dev_state, dev_a);

	cudaFree(dev_a);
	cudaFree(dev_state);

	return 0;
}
