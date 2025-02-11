
//xfail:BOOGIE_ERROR
//--blockDim=2 --gridDim=1 --no-inline
//Write by thread .+kernel.cu:8:4:
// to threadIdx.x != 0 we have 'data race'.

#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#define N 8 //2

__global__ void init_test(curandState *state, unsigned int *A) {
   curand_init(0, 0, 0, state);

   __syncthreads();

   A[threadIdx.x] =  curand(&state[threadIdx.x]);
//   if (threadIdx.x == 0) {
  //   A[0] = curand(state);
   //}
}

int main(){
	unsigned int *a;
	unsigned int *dev_a;
	curandState *dev_state; 

	int size = N*sizeof(unsigned int);

	a = (unsigned int*)malloc(size);
	cudaMalloc ((void**) &dev_a, size);

	printf("old a:  ");
	for (int i = 0; i < N; i++)
		printf("%u	", a[i]);

	cudaMalloc ( (void**) &dev_state, N*sizeof( curandState ) );

	// init_test<<<1,N>>>(dev_state, dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nnew a:  ");
	for (int i = 0; i < N; i++) {
		printf("%u	", a[i]);
		//assert((a[i] == 0 || a[i] == 1)); // we can't put assert() here because we get random numbers
		// maybe we can check if they are > 0 or not NULL... ?
	}

	free(a);
	cudaFree(&dev_a);
	cudaFree(&dev_state);

	return 0;
}

