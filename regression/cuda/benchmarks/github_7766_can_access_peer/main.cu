#include <cuda_runtime_api.h>
#include <assert.h>

int main() {
  int count, c;
  cudaGetDeviceCount(&count);
  assert(cudaDeviceCanAccessPeer(&c, 0, count) == cudaErrorInvalidDevice);
  assert(cudaDeviceCanAccessPeer(&c, -1, 0) == cudaErrorInvalidDevice);
  assert(cudaDeviceCanAccessPeer(&c, 0, 0) == cudaSuccess);
  assert(c == 0 || c == 1);
  return 0;
}
