#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define N 2

typedef float(*funcType)(float*, unsigned int);

__device__ float multiplyByTwo(float *v, unsigned int tid)
{
    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int tid)
{
    return v[tid] * 0.5f;
}

// Static pointers to device functions

	__device__ funcType p_mul_func = multiplyByTwo;

	__device__ funcType p_div_func = divideByTwo;

__global__ void foog(float *v, funcType f, funcType g, unsigned int size)
{
	assert(f == divideByTwo);
	assert(g == multiplyByTwo);

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    funcType h;
    
    if (f >= g)
      h = f;
    else
      h = g;

    if (tid < size)
    {
        v[tid] = (*h)(v, tid);
    }
}

int main(){
	float* w;
	float* dev_w;

	int size = N*sizeof(float);

	w =(float*) malloc(size);

	for (int i = 0; i < N; ++i){
		w[i] = i;
	}

	cudaMalloc((void**)&dev_w, size);

	cudaMemcpy(dev_w,w, size,cudaMemcpyHostToDevice);

	funcType host_f;

	cudaMemcpyFromSymbol( &host_f, &p_div_func, sizeof( funcType ), 0, cudaMemcpyDeviceToHost);

	funcType dev_f = host_f;

	funcType host_g;

	cudaMemcpyFromSymbol( &host_g, &p_mul_func, sizeof( funcType ), 0, cudaMemcpyDeviceToHost);

	funcType dev_g = host_g;

	// foog <<<1,N>>>(dev_w, dev_f, dev_g, N );

	cudaThreadSynchronize();

	cudaMemcpy(w,dev_w,size,cudaMemcpyDeviceToHost);

	printf("\nw:");
	for (int i = 0; i < N; ++i){
		printf(" %f	",	w[i]);
	}

	free(w);

	return 0;
}
