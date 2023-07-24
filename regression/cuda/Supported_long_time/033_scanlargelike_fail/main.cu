//fail
//--blockDim=32 --gridDim=64 --no-inline
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#define N 2//32

__device__ void f(float *odata, int* ai) {
    int thid = threadIdx.x;
    *ai = thid;
    odata[*ai] = 2*threadIdx.x;
}

__global__ void k(float *g_odata) {
    int ai;
    f(g_odata,&ai);
}


int main(){
	float *d;
	float *dev_d;

	d = (float*)malloc(N*sizeof(float));
	cudaMalloc ((void**) &dev_d, N*sizeof(float));

	cudaMemcpy(dev_d, d, N*sizeof(float),cudaMemcpyHostToDevice);

	//k <<<1,N>>>(dev_d);
	ESBMC_verify_kernel(k,1,N,dev_d);

	cudaMemcpy(d,dev_d,N*sizeof(float),cudaMemcpyDeviceToHost);

	printf("D: ");
	for (int i = 0; i < N; ++i) {
		printf(" %f		", d[i]);
		assert(!(d[i] == 2*i));
	}
	cudaFree(dev_d);

	return 0;
}
