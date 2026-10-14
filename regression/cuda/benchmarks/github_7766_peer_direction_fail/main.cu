#include <cuda_runtime_api.h>

__global__ void kernel(int *a) { a[threadIdx.x] = 1; }

// Peer access is one-way: device 0 reaching device 1 does not let device 1
// reach device 0's memory.
int main() {
  int *dev0;
  if (cudaSetDevice(0) != cudaSuccess)
    return 0;
  cudaMalloc((void **)&dev0, 2 * sizeof(int));
  if (cudaDeviceEnablePeerAccess(1, 0) != cudaSuccess)
    return 0;
  cudaSetDevice(1);
  ESBMC_verify_kernel(kernel, 1, 2, dev0);
  return 0;
}
