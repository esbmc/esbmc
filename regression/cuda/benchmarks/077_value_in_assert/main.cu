//pass: o caso deve passar por causa do assert que sempre é verdadeiro
//--blockDim=16 --gridDim=16 --no-inline
//a = 12
//b = 36
//c = 48

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#define N 2//16

__global__ void example(unsigned int a, unsigned int b, unsigned int c) {

    //__requires(a == 12);
    //__requires(b == 36);

    //__assert(a + b != c);
	assert(a + b != c);

}

int main(){

	unsigned int a, b, c;

	a=12;
	b=36;
	c=12;

	//example<<<N,N>>>(a,b,c);
	ESBMC_verify_kernel_u(example,1,N,a,b,c);

	cudaThreadSynchronize();
}
