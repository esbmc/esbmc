//pass
//--blockDim=64 --gridDim=64 --no-inline

#include <cuda_runtime_api.h>
#include <assert.h>
#define N 2//64

__global__ void foo(int* A)
{
 
 //__assert(__all(threadIdx.x < blockDim.x));
	assert(threadIdx.x < blockDim.x);

}

int main(){

	int *a,*dev_a;
	a = (int*)malloc(N*sizeof(int));
	
	cudaMalloc((void**)&dev_a,N*sizeof(int));
	cudaMemcpy(dev_a,a,N*sizeof(int),cudaMemcpyHostToDevice);

	//foo<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(foo,1,N,dev_a);
	
	cudaFree(dev_a);
	free(a);

}

