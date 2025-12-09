/******************************************* ternarytest2.cu ***************************************/
/*mostra 0 no índice 0, "c" no índice 1 e nos índices pares, mostra valor lixo nos demais índices */

#include <stdio.h>
#include "cuda_runtime_api.h"
#include <assert.h>
#define N 2 //64

__global__ void foo(float* A, float c) {

  A[threadIdx.x ? 2*threadIdx.x : 1] = c ;

}

int main()
{
	float* dev_b;
	float* b;
	float c = 2.0f;

	b = (float*)malloc(2*N*sizeof(float)); /* acessível apenas pela CPU função main e funções __host__ */

	cudaMalloc((void**)&dev_b, 2*N*sizeof(float)); /* acessível apenas pela GPU funções __global__ */

	//foo<<<1, N>>>(dev_b, c);
	ESBMC_verify_kernel_f(foo, 1, N, dev_b, c);

	cudaMemcpy(b, dev_b, 2*N*sizeof(float), cudaMemcpyDeviceToHost);

//	printf("\n");

	for (int i = 0; i < 2*N; ++i){
//	   printf("%f : ", b[i]);
	   if((i>0)&&(i%2==0))
		   assert(b[i] == c);
	}

	free(b);
	cudaFree(dev_b);
	return 0;
}
