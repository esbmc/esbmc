//pass
//--blockDim=512 --gridDim=1 --warp-sync=32 --no-inline

#include <cuda_runtime_api.h>
#include <stdio.h>

#define N 2


__global__ void scan (int* A)
{
	int tid = threadIdx.x;
	unsigned int lane = tid & 31;

	if (lane >= 1) A[tid] = A[tid - 1] + A[tid];
	if (lane >= 2) A[tid] = A[tid - 2] + A[tid];
	if (lane >= 4) A[tid] = A[tid - 4] + A[tid];
	if (lane >= 8) A[tid] = A[tid - 8] + A[tid];
	if (lane >= 16) A[tid] = A[tid - 16] + A[tid];
}

int main(){
	int *a;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = i;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	printf("old a:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", a[i]);

	//scan<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(scan, 1, 2, dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nnew a:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", a[i]);

	free(a);

	cudaFree(dev_a);


	return 0;
}
