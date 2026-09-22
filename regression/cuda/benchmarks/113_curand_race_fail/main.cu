
//xfail:BOOGIE_ERROR
//--blockDim=2 --gridDim=1 --no-inline
//Write by thread .+kernel\.cu:8:21:

#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 2 //4

__global__ void curand_test(curandState *state, float *A) { // test: replace curandState for curandStateXORWOW_t
   A[threadIdx.x] = curand_uniform(state);
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
