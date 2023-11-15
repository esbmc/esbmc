
//data race
//--blockDim=512 --gridDim=1 --warp-sync=32 --no-inline

#include <cuda_runtime_api.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#define N 4//512

__global__ void shuffle (int* A)
{
	int tid = threadIdx.x;
	int warp = tid / 2;//32;
	int* B = A + (warp*2);//32);
	A[tid] = B[(tid + 1)%2];//32];
}

int main() {
	int *a;
	int *b;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(N*size);
	b = (int*)malloc(N*size);

	for (int i = 0; i < N; i++)
		a[i] = i;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	printf("a:  ");

	for (int i = 0; i < N; i++)
		printf("%d        ", a[i]);

	//	shuffle<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(shuffle, 1, 2, dev_a);

	cudaMemcpy(b,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nFunction Results:\n   ");

	for (int i = 0; i < N; i++){
		printf("%d        ", b[i]);
		assert(b[i]==a[i+1]);
	}
	free(a);free(b);

	cudaFree(dev_a);

	return 0;
}
