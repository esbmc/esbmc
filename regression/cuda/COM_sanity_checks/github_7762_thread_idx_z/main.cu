#include <cuda_runtime_api.h>

/* 2 blocks x 1 thread: blockDim is (1,1,1), so threadIdx.z is 0 for every
   thread and only a[0] is written (#7762). */
__global__ void kernel(int *a) { a[threadIdx.z] = 1; }

int main()
{
  int *dev_a;
  cudaMalloc((void **)&dev_a, 1 * sizeof(int));
  ESBMC_verify_kernel(kernel, 2, 1, dev_a);
  cudaFree(dev_a);
  return 0;
}
