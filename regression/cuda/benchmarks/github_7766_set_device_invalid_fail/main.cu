#include <cuda_runtime_api.h>
#include <assert.h>

int main() {
  int count;
  cudaGetDeviceCount(&count);
  assert(cudaSetDevice(-1) == cudaErrorInvalidDevice);
  assert(cudaSetDevice(count) == cudaSuccess);
  return 0;
}
