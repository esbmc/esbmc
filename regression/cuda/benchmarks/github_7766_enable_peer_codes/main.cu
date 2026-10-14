#include <cuda_runtime_api.h>
#include <assert.h>

int main() {
  int count;
  cudaGetDeviceCount(&count);
  if (cudaSetDevice(1) != cudaSuccess)
    return 0;
  assert(cudaDeviceEnablePeerAccess(0, 1) == CUDA_ERROR_INVALID_VALUE);
  assert(cudaDeviceEnablePeerAccess(1, 0) == cudaErrorInvalidDevice);
  assert(cudaDeviceEnablePeerAccess(-1, 0) == cudaErrorInvalidDevice);
  assert(cudaDeviceEnablePeerAccess(count, 0) == cudaErrorInvalidDevice);
  cudaError_t r = cudaDeviceEnablePeerAccess(0, 0);
  assert(r == cudaSuccess || r == cudaErrorInvalidDevice);
  if (r == cudaSuccess)
    assert(
      cudaDeviceEnablePeerAccess(0, 0) ==
      CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
  return 0;
}
