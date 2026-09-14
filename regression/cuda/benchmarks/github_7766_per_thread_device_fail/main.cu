#include <cuda_runtime_api.h>
#include <pthread.h>

__global__ void kernel(int *a) { a[threadIdx.x] = 1; }

int *dev0;

void *other(void *) {
  if (cudaSetDevice(1) == cudaSuccess)
    ESBMC_verify_kernel(kernel, 1, 2, dev0);
  return NULL;
}

int main() {
  pthread_t t;
  if (cudaSetDevice(0) != cudaSuccess)
    return 0;
  cudaMalloc((void **)&dev0, 2 * sizeof(int));
  pthread_create(&t, NULL, other, NULL);
  pthread_join(t, NULL);
  return 0;
}
