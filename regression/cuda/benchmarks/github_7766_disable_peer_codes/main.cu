#include <cuda_runtime_api.h>
#include <assert.h>

int main() {
  int count;
  cudaGetDeviceCount(&count);
  if (cudaSetDevice(1) != cudaSuccess)
    return 0;
  assert(cudaDeviceDisablePeerAccess(-1) == cudaErrorInvalidDevice);
  assert(cudaDeviceDisablePeerAccess(count) == cudaErrorInvalidDevice);
  assert(cudaDeviceDisablePeerAccess(0) == CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
  return 0;
}
