#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <assert.h>
static const int WORK_SIZE = /*256*/ 2;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */


__device__ unsigned int bitreverse1(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse1(idata[threadIdx.x]);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main() {
	void *d = NULL;
	int i;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (i = 0; i < WORK_SIZE; i++){
		idata[i] = (unsigned int) i+1;
		printf("%u; ", idata[i]);
	}

	printf("\n");

	cudaMalloc((void**) &d, sizeof(int) * WORK_SIZE);
	cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice);

	//	bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);
	ESBMC_verify_kernel(bitreverse, 1, WORK_SIZE /* *sizeof(int)*/, d);

	cudaThreadSynchronize();	// Wait for the GPU launched work to complete
	cudaGetLastError();
	cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost);

	for (i = 0; i < WORK_SIZE; i++){
		printf("Input value: %u, device output: %u\n", idata[i], odata[i]);
		assert((idata[i]==1)and(odata[i]==128));
	}
	cudaFree((void*) d);
	cudaDeviceReset();

	return 0;
}

