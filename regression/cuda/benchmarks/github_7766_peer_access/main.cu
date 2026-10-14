#include <cuda_runtime_api.h>

__global__ void kernel(int *a) { a[threadIdx.x] = 1; }

int main() {
  int *dev0;
  cudaSetDevice(0);
  cudaMalloc((void **)&dev0, 2 * sizeof(int));
  if (cudaSetDevice(1) != cudaSuccess)
    return 0;
  if (cudaDeviceEnablePeerAccess(0, 0) != cudaSuccess)
    return 0;
  ESBMC_verify_kernel(kernel, 1, 2, dev0);
  cudaFree(dev0);
  return 0;
}
