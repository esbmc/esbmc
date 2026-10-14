#include <cuda_runtime_api.h>
#include <pthread.h>
#include <assert.h>

void *other(void *) {
  int n;
  cudaGetDeviceCount(&n);
  return NULL;
}

int main() {
  pthread_t t;
  int n, m;
  pthread_create(&t, NULL, other, NULL);
  cudaGetDeviceCount(&n);
  cudaGetDeviceCount(&m);
  assert(n == 1);
  pthread_join(t, NULL);
  return 0;
}
