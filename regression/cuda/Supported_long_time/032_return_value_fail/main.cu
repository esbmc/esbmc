//fail:assertion
//--blockDim=1024 --gridDim=1 --no-inline

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#define N 2//1024

typedef float(*funcType)(float*, unsigned int);

__device__ float multiplyByTwo(float *v, unsigned int tid)
{
    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int tid)
{
    return v[tid] * 0.5f;
}

__device__ funcType grabFunction(int i) {
  //__requires(i != 0);
  //__ensures(__return_val_funptr(funcType) == divideByTwo);
  if (i == 0)
    return multiplyByTwo;
  else
    return divideByTwo;
}

__global__ void foo(float *v, unsigned int size, int i)
{
    //__requires(i != 0);
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    funcType f = grabFunction(i);

    if (tid < size)
    {
    	float x = (*f)(v, tid);
		x += multiplyByTwo(v, tid);
		v[threadIdx.x] = x;
    }
}

int main(){

	int c = 0; /*define se multiplicará ou dividirá por 2, deve ser 0 ou outro valor, para escolher a função*/
	float* v;
	float* a;
	float* dev_v;
	funcType fun;

	//fun = (funcType)malloc(sizeof(funcType));

	printf("Digite 0 para multiplicar um vetor por 4 ou\nDigite outro valor para multiplicar um vetor por 2.5: \n");
	scanf("%u", &c);

	v = (float*)malloc(N*sizeof(float));
	a = (float*)malloc(N*sizeof(float));

	for (int i = 0; i < N; ++i){
		v[i] = rand() %10+1;
		printf(" %.1f; ", v[i]);
	}

	printf("\n");

	cudaMalloc((void**)&dev_v, N*sizeof(float));

	cudaMemcpy(dev_v, v, N*sizeof(float), cudaMemcpyHostToDevice);

		//foo<<<1, N>>>(dev_v, N, c);
		ESBMC_verify_kernel_fuintint (foo,1, N, dev_v, N, c);

	cudaMemcpy(a, dev_v, N*sizeof(float), cudaMemcpyDeviceToHost);


	for (int i = 0; i < N; ++i){
		printf(" %.1f; ", a[i]);
		if (c==0)
			assert(a[i]!=4*v[i]);
		else
			assert(a[i]!=2.5*v[i]);
	}

	free(v); free(a);
	cudaFree(dev_v);

   return 0;
}
