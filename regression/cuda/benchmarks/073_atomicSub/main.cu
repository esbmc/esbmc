//pass
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda_runtime_api.h>
#include <stdio.h>

#define N 2 //1024

__global__ void definitions (int* A, unsigned int* B)
{
	atomicSub(A,10);

	atomicSub(B,5);

}

int main (){

	int a = 5;
	int *dev_a;

	cudaMalloc ((void**) &dev_a, sizeof(int));

	cudaMemcpy(dev_a, &a, sizeof(int),cudaMemcpyHostToDevice);

	unsigned int b = 10;
	unsigned int *dev_b;

	cudaMalloc ((void**) &dev_b, sizeof(unsigned int));

	cudaMemcpy(dev_b, &b, sizeof(unsigned int),cudaMemcpyHostToDevice);

	//definitions <<<1,N>>>(dev_a,dev_b);
	ESBMC_verify_kernel(definitions,1,N,dev_a,dev_b);

	cudaMemcpy(&a,dev_a,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(&b,dev_b,sizeof(unsigned int),cudaMemcpyDeviceToHost);

	printf("A: %d\n", a);
	printf("B: %u\n", b);

	assert(a==-15);
	assert(b==0);

	cudaFree(dev_a);
	cudaFree(dev_b);

	return 0;

}
