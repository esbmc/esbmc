//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1 --warp-sync=16 --no-inline
//It should show only the values from B[0] to B[31], but it exceeds.

#include <cuda_runtime_api.h>
#include <stdio.h>

#define N 2//32//1024

__global__ void shuffle (int* A)
{
	int tid = threadIdx.x;
	int warp = tid / 32;
	int* B = A + (warp*32);
	A[tid] = B[(tid + 1)%32];
}

int main() {
	int *a;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(N*size);

	for (int i = 0; i < N; i++)
		a[i] = i;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	printf("a:  ");

	for (int i = 0; i < N; i++)
		printf("%d        ", a[i]);

    // shuffle<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(shuffle, 1, N, dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nFunction Results:\n   ");

	for (int i = 0; i < N; i++)
		printf("%d        ", a[i]);

	free(a);

	cudaFree(dev_a);

	return 0;
}
