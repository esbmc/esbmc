
//xfail:BOOGIE_ERROR
//--blockDim=2 --gridDim=1 --no-inline
//Write by thread .+kernel\.cu:8:21:

#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>

#define N 4

__global__ void curand_test(curandState *state, float *A) { // test: replace curandState for curandStateXORWOW_t
   A[threadIdx.x] = curand_uniform(state);
}

int main(){
	float *a;
	float *dev_a;
	curandState *dev_state; 
	
	int size = N*sizeof(float);

	a = (float*)malloc(size);
	cudaMalloc ((void**) &dev_a, size);

	printf("old a:  ");
	for (int i = 0; i < N; i++)
		printf("%f	", a[i]);

	cudaMalloc ( (void**) &dev_state, N*sizeof( curandState ) );

	//curand_test<<<1,N>>>(dev_state, dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nnew a:  ");
	for (int i = 0; i < N; i++) {
		printf("%f	", a[i]);
		assert(a[i] == a[i+1]);
	}

	free(a);
	cudaFree(&dev_a);
	cudaFree(&dev_state);

	return 0;
}
