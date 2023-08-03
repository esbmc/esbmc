//pass
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda_runtime_api.h>
#include <stdio.h>

#define N 2 //1024

__global__ void definitions (unsigned int* B)
{
  atomicInc(B,7);//0111 -> 1000 -> 0000 -> 0001 -> 0010 -> 0011 -> 0100 -> 0101 -> 0110 ...
  	  /*the second argument on atomicInc() is a limit for increments. When this limit is reached, B receives 0*/
}

int main (){

	unsigned int b = 5;
	unsigned int *dev_b;

	cudaMalloc ((void**) &dev_b, sizeof(unsigned int));

	cudaMemcpy(dev_b, &b, sizeof(unsigned int),cudaMemcpyHostToDevice);

//	definitions <<<1,N>>>(dev_b);
	ESBMC_verify_kernel_ui(definitions,1,N,dev_b);
	cudaMemcpy(&b,dev_b,sizeof(unsigned int),cudaMemcpyDeviceToHost);

	printf("B: %u\n", b);

	assert(b==7);

	cudaFree(dev_b);
	return 0;

}
