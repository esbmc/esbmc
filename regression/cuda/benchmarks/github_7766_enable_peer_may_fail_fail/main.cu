#include <cuda_runtime_api.h>
#include <assert.h>

// Whether a device can reach its peer is platform-dependent, so enabling
// peer access must be able to fail.
int main() {
  if (cudaSetDevice(1) != cudaSuccess)
    return 0;
  assert(cudaDeviceEnablePeerAccess(0, 0) == cudaSuccess);
  return 0;
}
