#include <cuda_runtime_api.h>
#include <assert.h>

__global__ void kernel(int *a) { a[threadIdx.x] = 1; }

int main() {
  int count;
  int *dev;
  cudaGetDeviceCount(&count);
  cudaMalloc((void **)&dev, 2 * sizeof(int));

  assert(cudaSetDevice(-1) == cudaErrorInvalidDevice);
  assert(cudaSetDevice(count) == cudaErrorInvalidDevice);

  ESBMC_verify_kernel(kernel, 1, 2, dev);

  cudaFree(dev);
  return 0;
}
