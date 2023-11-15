//pass
//--blockDim=64 --gridDim=64 --no-inline
#include <cuda_runtime_api.h>

inline __device__ void f() __attribute__((always_inline));
inline __device__ void f() {
}

inline __device__ void g() __attribute__((always_inline));
inline __device__ void g() {
  f();
}

__global__ void k() {
  g();
}


int main(){

	//k<<<2,2>>>();
	ESBMC_verify_kernel(k,1,2);

	cudaThreadSynchronize();
}
