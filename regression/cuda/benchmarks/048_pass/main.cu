//pass
//--blockDim=64 --gridDim=64 --no-inline
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime_api.h"

#define N 2//64


__global__ void foo() {

  float x = (float)2;

}

int main(){
	
	//foo<<<1, N>>>();
	ESBMC_verify_kernel(foo,1,N);

	cudaThreadSynchronize();	

	return 0;

}
