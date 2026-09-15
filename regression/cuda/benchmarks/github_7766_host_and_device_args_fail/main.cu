#include <cuda_runtime_api.h>
#include <stdlib.h>

__global__ void kernel(int *a, int *b, int *c) {
  a[threadIdx.x] = b[threadIdx.x] + c[threadIdx.x];
}

int main() {
  int host[2] = {1, 2};
  int *dev0;
  int *dev1;
  cudaSetDevice(0);
  cudaMalloc((void **)&dev0, 2 * sizeof(int));
  if (cudaSetDevice(1) != cudaSuccess)
    return 0;
  cudaMalloc((void **)&dev1, 2 * sizeof(int));
  ESBMC_verify_kernel(kernel, 1, 2, dev1, host, dev0);
  return 0;
}
