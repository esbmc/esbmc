#include <cuda_runtime_api.h>
#include <pthread.h>

__global__ void kernel(int *a) { a[threadIdx.x] = 1; }

void *other(void *) {
  cudaSetDevice(1);
  return NULL;
}

int main() {
  pthread_t t;
  int *dev0;
  if (cudaSetDevice(0) != cudaSuccess)
    return 0;
  cudaMalloc((void **)&dev0, 2 * sizeof(int));
  pthread_create(&t, NULL, other, NULL);
  ESBMC_verify_kernel(kernel, 1, 2, dev0);
  pthread_join(t, NULL);
  return 0;
}
