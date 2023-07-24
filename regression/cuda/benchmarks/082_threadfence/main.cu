//pass
//--blockDim=32 --gridDim=2

#include <cuda_runtime_api.h>

__global__ void foo() {
    __threadfence();
}

int main(){

	//foo<<<1,2>>>();
	ESBMC_verify_kernel(foo,1,2);

	cudaThreadSynchronize();

	return 0;
}
