//fail
//--blockDim=64 --gridDim=64 --no-inline
#include <stdio.h>
#include "cuda_runtime_api.h"
#define N 1

__device__ void bar (int *p){

    int a = 0;

    p = &a;
}

__global__ void foo (int* p, int* q){

    if (*p > 10){
        bar(p);
        //*p = 23; // remove this comment to see that the __device__ function does not work
    }
    else {
        bar(q);
        //*q = 23; // remove this comment to see that the __device__ function does not work
    }
}

int main(){
	int *a, *b;
	int *dev_a, *dev_b;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);

	a = (int*)malloc(size);
	b = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 1;

	for (int i = 0; i < N; i++)
		b[i] = 1;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,size, cudaMemcpyHostToDevice);

	printf("a:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", a[i]);

	printf("\nb:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", b[i]);

	// foo<<<1,2>>>(dev_a,dev_b);
	ESBMC_verify_kernel(foo,1,2,dev_a,dev_b);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(b,dev_b,size,cudaMemcpyDeviceToHost);

	printf("\nnew a and new b:\n");

	printf("a:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", a[i]);

	printf("\nb:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", b[i]);

	assert(b[0]!=1);

	free(a); free(b);

	cudaFree(dev_a);
	cudaFree(dev_b);

	return 0;
}
