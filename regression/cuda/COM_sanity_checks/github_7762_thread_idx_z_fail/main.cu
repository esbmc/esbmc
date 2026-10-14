#include <cuda_runtime_api.h>
#include <assert.h>

/* As github_7762_thread_idx_z: both threads write a[0], so a[1] is never set
   and this assertion must fail. It held while thread 1 got threadIdx.z == 1. */
__global__ void kernel(int *a) { a[threadIdx.z] = blockIdx.x + 1; }

int main()
{
  int *dev_a;
  cudaMalloc((void **)&dev_a, 2 * sizeof(int));
  ESBMC_verify_kernel(kernel, 2, 1, dev_a);
  assert(dev_a[1] == 2);
  cudaFree(dev_a);
  return 0;
}
