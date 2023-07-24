//pass
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda_runtime_api.h>
#include <stdio.h>

#define N 2 //1024

__global__ void definitions (unsigned int* B)
{
  atomicDec(B,7);//0111 -> 1000 -> 0000 -> 0001 -> 0010 -> 0011 -> 0100 -> 0101 -> 0110 ...
  	  /*the second argument on atomicDec() is a limit for decs. When this limit is reached, B receives <LIM>*/
}

int main (){

	unsigned int b = 5;
	unsigned int *dev_b;

	cudaMalloc ((void**) &dev_b, sizeof(unsigned int));

	cudaMemcpy(dev_b, &b, sizeof(unsigned int),cudaMemcpyHostToDevice);

	//definitions <<<1,N>>>(dev_b);
	ESBMC_verify_kernel_ui(definitions,1,N,dev_b);

	cudaMemcpy(&b,dev_b,sizeof(unsigned int),cudaMemcpyDeviceToHost);

	printf("B: %u\n", b);
	assert(b==3);

	cudaFree(dev_b);
	return 0;

}
