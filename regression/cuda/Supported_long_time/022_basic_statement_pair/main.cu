//pass: checka a declaração de uma variável do tipo "ponteiro pra função" dentro do kernel
//--blockDim=1024 --gridDim=1 --no-inline
#include "cuda_runtime_api.h"
#include <stdio.h>
#include <assert.h>

#define tid (blockIdx.x * blockDim.x + threadIdx.x)
#define N 2//1024

__device__ void multiplyByTwo(float *v, unsigned int index)
{  
	v[index] = v[index] * 2.0f;
}

__device__ void divideByTwo(float *v, unsigned int index)
{  
    v[index] = v[index] * 0.5f;
}

typedef void(*funcType)(float*, unsigned int);

__global__ void foo(float *v, unsigned int i)
{
    	assert(i == 1 | i == 2);
    	
    funcType f;

    if (i == 1)
      f = multiplyByTwo;
    else if (i == 2)
      f = divideByTwo;
    else
      f = NULL;

    (*f)(v, tid);
}

int main(){

	unsigned int c = 1; /*define se multiplicará ou dividirá por 2, deve ser 1 ou 2, para escolher a função*/
	float* v;
	float* a;
	float* dev_v;

	v = (float*)malloc(N*sizeof(float));
	a = (float*)malloc(N*sizeof(float));

	for (int i = 0; i < N; ++i){
		v[i] = rand() %10+1;
		printf(" %.1f; ", v[i]);
	}

	printf("\n");

	cudaMalloc((void**)&dev_v, N*sizeof(float));

	cudaMemcpy(dev_v, v, N*sizeof(float), cudaMemcpyHostToDevice);

		//foo<<<1, N>>>(dev_v, c);
		ESBMC_verify_kernel_fuintt(foo, 1, N, dev_v, c);

	cudaMemcpy(a, dev_v, N*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i){
		printf(" %.1f; ", a[i]);
		if (c==1)
			assert(a[i]==2*v[i]);
		else if (c==2)
			assert(a[i]==0.5*v[i]);
	}

	free(v); free(a);
	cudaFree(dev_v);

	return 0;
}
