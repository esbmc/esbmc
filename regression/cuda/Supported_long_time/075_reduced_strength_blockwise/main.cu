//pass
//--blockDim=256 --gridDim=2 -DWIDTH=2064 --no-inline
#include <cuda_runtime_api.h>
#include <stdio.h>

#define GRIDDIM 1
#define BLOCKDIM 2//256
#define WIDTH 2//2048
#define N WIDTH
/*
 * This kernel demonstrates a blockwise strength-reduction loop.
 * Each block is given a disjoint partition (of length WIDTH) of A.
 * Then each thread writes multiple elements in the partition.
 * It is not necessarily the case that WIDTH%blockDim.x == 0
 */

__global__ void k(int *A) {

  for (int i=threadIdx.x; i<WIDTH; i+=blockDim.x) {

    A[blockIdx.x*WIDTH+i] = i;
  }
}

int main (){
	int *a;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 0;

	cudaMemcpy(dev_a,a,size,cudaMemcpyHostToDevice);

	//k <<<GRIDDIM, BLOCKDIM>>>(dev_a);
	ESBMC_verify_kernel(k,GRIDDIM,BLOCKDIM,dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++){
		assert(a[i]== i);
	}

	free(a);
	cudaFree(dev_a);
	return 0;
}
