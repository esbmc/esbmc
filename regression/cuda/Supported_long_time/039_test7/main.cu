/*** substitui os valores aleatórios de determinado vetor de tamanho N por valores ordenados de 0 a N ***/
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#define N 2//64

__global__ void foo(int* glob) {

  int a;

  int* p;

  a = 0;

  p = &a;

  *p = threadIdx.x;

  glob[*p] = threadIdx.x;
}

int main(){

	int* v;
	int* dev_v;

	/* seta o tamanho de v e inicia com com valores randômicos */
	v = (int*)malloc(N*sizeof(int)); /* acessível apenas pela CPU função main e funções __host__ */

	for (int i = 0; i < N; ++i)
		v[i] = rand() %20+1;;

	for (int i = 0; i < N; ++i)
		printf(" %d    :", v[i]);

	cudaMalloc((void**)&dev_v, N*sizeof(int)); /* acessível apenas pela GPU funções __global__ */

	cudaMemcpy(dev_v, v, N*sizeof(int), cudaMemcpyHostToDevice);

	//foo<<<1, N>>>(dev_v);
	ESBMC_verify_kernel(foo,1,N,dev_v);

	cudaMemcpy(v, dev_v, N*sizeof(int), cudaMemcpyDeviceToHost);

	printf("\n\n\n");

	for (int i = 0; i < N; ++i){
		printf(" %d    :", v[i]);
		assert(v[i]==i);
	}
	free(v);
	cudaFree(dev_v);

	return 0;
}
