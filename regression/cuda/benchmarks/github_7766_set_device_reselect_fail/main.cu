#include <cuda_runtime_api.h>
#include <assert.h>

int main() {
  assert(cudaSetDevice(0) == cudaSuccess);
  assert(cudaSetDevice(0) == cudaErrorDeviceAlreadyInUse);
  return 0;
}
