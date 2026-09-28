#include <cuda_runtime_api.h>

__global__ void kernel(int *a) { a[threadIdx.x] = 1; }

int main() {
  int *dev1;
  cudaSetDevice(1);
  cudaMalloc((void **)&dev1, 2 * sizeof(int));

  ESBMC_verify_kernel(kernel, 1, 2, dev1);

  cudaFree(dev1);
  return 0;
}
