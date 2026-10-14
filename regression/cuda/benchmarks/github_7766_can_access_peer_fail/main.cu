#include <cuda_runtime_api.h>
#include <assert.h>

int main() {
  int c;
  assert(cudaDeviceCanAccessPeer(&c, 0, 0) == cudaSuccess);
  assert(c == 1);
  return 0;
}
