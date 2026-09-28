#include <cuda_runtime_api.h>
#include <assert.h>

int main() {
  if (cudaSetDevice(1) != cudaSuccess)
    return 0;
  assert(cudaDeviceDisablePeerAccess(-1) == CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
  return 0;
}
