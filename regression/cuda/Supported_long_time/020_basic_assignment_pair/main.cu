#include <stdio.h>
#include "cuda_runtime_api.h"
#define N 2 //64

__device__ float multiplyByTwo(float *v, unsigned int tid) {

    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int tid) {

    return v[tid] * 0.5f;
}

typedef float(*funcType)(float*, unsigned int);

__global__ void foor(float *v, unsigned int size, unsigned int i) {

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    funcType f;

	/*** ESBMC_assert (i = 1 || i = 2); ***/
		assert(i == 1 || i == 2);
		
    if (i == 1)
      f = multiplyByTwo;
    else if (i == 2)
      f = divideByTwo;
    else
      f = NULL;

    if (tid < size)
    {
        float x = (*f)(v, tid);
        x += multiplyByTwo(v, tid);
		v[tid] = x;
    }
}

int main(){

	unsigned int c = 1; /* c defines which function will be selected (multiplyByTwo or divideByTwo), it must be 1 or 2 for choose the function */
	float* v;
	float* dev_v;

	/* sets the size of v */
	v = (float*)malloc(N*sizeof(float)); /* visible only by CPU: function main() and __host__ functions*/

	for (int i = 0; i < N; ++i)
		v[i] = i;

	for (int i = 0; i < N; ++i)
		printf(" %f    :", v[i]);

	cudaMalloc((void**)&dev_v, N*sizeof(float)); /* visible only by GPU: __global__ functions */

	cudaMemcpy(dev_v, v, N*sizeof(float), cudaMemcpyHostToDevice);	

	//foor<<<1, N>>>(dev_v, N, c);
	ESBMC_verify_kernel_fuintt(foor, 1,N, dev_v, N, c);

	cudaMemcpy(v, dev_v, N*sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("\n");

	for (int i = 0; i < N; ++i) {
		printf(" %f    :", v[i]);
		if (c == 1)
			assert(v[i] == 4*i);
		else
			assert(v[i] == 2.5*i);
	}

	free(v);
	cudaFree(dev_v);

	return 0;
}
