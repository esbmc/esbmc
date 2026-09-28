/* Reduced from benchmarks/003_add_one_pair: kernel args round-trip
 * through the OM's global argument struct, and on builds whose
 * simplifier folds those member reads the path-guard chains activate
 * and mis-decide a bookkeeping guard — the assert then races ahead of
 * the kernel stores and reads uninitialized memory. Verdicts split
 * per build environment on the same source; this pins SUCCESSFUL. */
#include <cuda_runtime_api.h>
#include <assert.h>
__global__ void race_test (unsigned int* i, int* A)
{
  int j = *i;
  *i = j + 1;
  A[j] = 0;
}
int main(){
	unsigned int *dev_i;
	int *dev_A;
	cudaMalloc((void**)&dev_A, 2*sizeof(int));
	cudaMalloc((void**)&dev_i, sizeof(unsigned int));
	*dev_i = 0;
	ESBMC_verify_kernel_u(race_test,1,2,dev_i,dev_A);
	assert(dev_A[0]==0 || dev_A[0]==1);
	return 0;
}
