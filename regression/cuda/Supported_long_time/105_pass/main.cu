//pass
//--blockDim=1024 --gridDim=1
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <string.h>
#include <assert.h>

#define N 4//1024

//swap the strings
__device__ void swap(char *in, char *out) {
	char tmp[N];
	tmp[threadIdx.x]= in[threadIdx.x];
	__syncthreads();
	in[threadIdx.x] = out[threadIdx.x];
	__syncthreads();
	out[threadIdx.x]= tmp[threadIdx.x];
}

__global__ void foo(char *A, char *B, char* c)
{
  char *choice1 = A;	//It Makes choice1 receives A
  char *choice2 = B;	//It Makes choice2 receives B
  swap(choice1, choice2);		//This function swaps choice1 and choice2
	assert(strcmp(choice1,choice2) == 0);
}

int main() {

	char *a;
	char *b;
	
	char *dev_a;
	char *dev_b;
	char* dev_c;

	int size = N*sizeof(char);

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, sizeof(char));

	a = (char*)malloc(size);
	b = (char*)malloc(size);

	strcpy(a, "123");
	strcpy(b, "123");

	assert(strcmp(a,b) == 0);

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,size, cudaMemcpyHostToDevice);

	//foo<<<1,N>>>(dev_a, dev_b, dev_c);
	ESBMC_verify_kernel_c(foo, 1, 2, dev_a, dev_b, dev_c);

	char *d;
	char *e;
	d = (char*)malloc(size);
	e = (char*)malloc(size);

	cudaMemcpy(d,dev_a,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(e,dev_b,size,cudaMemcpyDeviceToHost);

	free(a); free(b); free(d); free(e);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
