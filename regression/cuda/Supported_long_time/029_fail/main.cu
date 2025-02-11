#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>
#include <assert.h>

#define N 2//(64*64)//(2048*2048)
#define THREADS_PER_BLOCK 2//512

__global__ void Asum(int *a, int *b, int *c){
	int index = threadIdx.x;
	c[index] = a[index] + b[index];
}

int main(void){
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c,size);

	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 10;

	for (int i = 0; i < N; i++)
		b[i] = 10;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,size, cudaMemcpyHostToDevice);

	printf("a:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", a[i]);

	printf("\nb:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", b[i]);

	//Asum<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dev_a,dev_b,dev_c);
	ESBMC_verify_kernel(Asum, N/THREADS_PER_BLOCK,THREADS_PER_BLOCK,dev_a,dev_b,dev_c);

	cudaMemcpy(c,dev_c,size,cudaMemcpyDeviceToHost);

	printf("\nResultado da soma de a e b eh:\n   ");

	for (int i = 0; i < N; i++){
		printf("%d	", c[i]);
		assert(c[i]!=a[i]+b[i]);
	}

	free(a); free(b); free(c);

	cudaFree(dev_a);
	cudaFree(dev_c);
	cudaFree(dev_b);

	return 0;

}
