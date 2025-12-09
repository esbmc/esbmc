#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <assert.h>

#define N 16

__device__ int index(int col, int row, int ord){
	return (row *ord)+col;
}

__global__ void Transpose(int *c, const int *a){
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
	int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    c[index(row,col,4)] = a[index(col, row, 4)] ;
}

int main()
{
    const int arraySize = 16;
    const int a[arraySize] = { 1, 2, 3, 4, 5 ,6,7,8,9,10,11,12,13,14,15,16};
    int c[arraySize] = { 0 };

    int *dev_a = 0;
    int *dev_c = 0;

    // Allocate GPU buffers for three vectors (one input, one output)    .
        cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
        cudaMalloc((void**)&dev_a, arraySize * sizeof(int));

        // Copy input vectors from host memory to GPU buffers.
        cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_c, c, arraySize * sizeof(int), cudaMemcpyHostToDevice);

        // Launch a kernel on the GPU with one thread for each element.
    	dim3 dimgrid(2, 2);
    	dim3 dimblock(2, 2);
    	//Transpose<<<dimgrid, dimblock>>>(dev_c, dev_a);
	    ESBMC_verify_kernel(Transpose,1,2,dev_c,dev_a);

        // Copy output vector from GPU buffer to host memory.
        cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(dev_c);
        cudaFree(dev_a);

	for (int i = 0; i < arraySize; i++){
		printf("%d ",c[i]);
		if(i<3)
			assert(c[i+1]!=c[i]+4);

	}

    return 0;
}
