//pass
//--blockDim=2 --gridDim=2
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define N 2

struct S {
  int * p;
};

__global__ void foo(int * A) {

  S myS;
  myS.p = A;
  int * q;
  q = myS.p;
  q[threadIdx.x + blockDim.x*blockIdx.x] = threadIdx.x;

}

int main() {
	int *a;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 5;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	//foo<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(foo,1,N,dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		assert(a[i] == 0 || a[i] == 1);
	}

	free(a);

	cudaFree(dev_a);

	return 0;
}
