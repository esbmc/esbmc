#include <cuda_runtime_api.h>
#include <assert.h>

int main() {
  int count;
  assert(cudaGetDeviceCount(&count) == cudaSuccess);
  assert(count >= 1);

  assert(cudaSetDevice(0) == cudaSuccess);
  assert(cudaSetDevice(0) == cudaSuccess);
  return 0;
}
