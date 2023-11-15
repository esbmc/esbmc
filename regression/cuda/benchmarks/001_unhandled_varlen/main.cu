//xfail:BUGLE_ERROR
//--gridDim=1 --blockDim=32 --no-inline

//This kernel is racy: memset is called with variable length.
//#define memset(dst,val,len) __builtin_memset(dst,val,len)

#define N 2//32

#include <stdio.h>
#include <cuda_runtime_api.h>

__device__ int bar(void){
	int value;
	return value;
}

__global__ void kernel(uint4 *out) {
  uint4 vector;
  int len = bar();
   memset(&vector, 5, len); /*modify manually the value of len to see the bugs*/
  out[threadIdx.x] = vector;
}

int main(){
	uint4 *a;
	uint4 *dev_a;
	int size = N*sizeof(uint4);

	a = (uint4*)malloc(size);

	/* initialization of a */
	for (int i = 0; i < N; i++) {
		a[i].x = i; a[i].y = i; a[i].z = i, a[i].w = i;
	}

	cudaMalloc((void**)&dev_a, size);

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	// kernel<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(kernel,1,N,dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	free(a);
	return 0;
}
