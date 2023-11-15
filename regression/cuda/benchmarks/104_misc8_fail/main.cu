//fail
//--blockDim=1024 --gridDim=1

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define N 2//1024

//replace out with in
__device__ void bar(char **in, char **out) {
	char tmp = (*in)[threadIdx.x];
	  out[0][threadIdx.x] = tmp;
	  *out = *in;
}

__global__ void foo(char *A, char *B, char* c)
{
  char *choice1 = *c ? A : B;	//It Makes choice1 receives A
  char *choice2 = *c ? B : A;	//It Makes choice2 receives B

  bar(&choice1, &choice2);
  bar(&choice1, &choice2);
}

int main() {

	char *a;
	char *d;
	char *dev_a;
	char *b;
	char *e;
	char *dev_b;
	char c = 'x';
	char* dev_c;
	int size = N*sizeof(char);

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, sizeof(char));

	a = (char*)malloc(size);
	d = (char*)malloc(size);
	b = (char*)malloc(size);
	e = (char*)malloc(size);

	strcpy(a, "CudaEsbmc Test1");
	strcpy(b, "CudaEsbmc Test2");

	assert(strcmp(a,b) != 0);

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c,&c,sizeof(char), cudaMemcpyHostToDevice);

	printf("\nNa CPU: \n");
	printf(a); printf("\n");
	printf(b); printf("\n");

	printf("\nNa GPU:\n");

	//foo<<<1,N>>>(dev_a, dev_b, dev_c);
	ESBMC_verify_kernel_c(foo, 1, N, dev_a, dev_b, dev_c);

	printf("\n");
	cudaMemcpy(d,dev_a,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(e,dev_b,size,cudaMemcpyDeviceToHost);

	printf(d); printf("\n");
	printf(e); printf("\n");

	free(a);free(d);	free(b);free(e);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
