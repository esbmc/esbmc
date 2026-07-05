// Regression test: calloc(2^30, sizeof(int)) in 32-bit mode previously
// triggered a false alarm because nmemb*size wrapped to 0, malloc(0) was
// called, and data[0] got a nondet (non-zero) value.
//
// After the overflow fix, __ESBMC_assume(nmemb <= SIZE_MAX/size) prunes
// the n=2^30 path entirely: the assertion is never evaluated on the
// overflow path, so no false positive fires.  ESBMC should report
// VERIFICATION SUCCESSFUL (no counterexample found).
#include <stdlib.h>
#include <limits.h>

void reach_error(void) {}
static void verifier_assert(int cond)
{
  if (!cond)
    reach_error();
}

int main(void)
{
  // n = 2^30.  On 32-bit size_t: n * sizeof(int) = 4294967296 = 0 (overflow).
  // Previously this bypassed memset and gave nondet data[0].
  int n = 1073741824;
  int *data = (int *)calloc((size_t)n, sizeof(int));
  // If calloc succeeds, data must be zeroed.  The overflow path is pruned,
  // so ESBMC will not explore this path -> VERIFICATION SUCCESSFUL.
  if (data)
    verifier_assert(data[0] == 0);
  free(data);
  return 0;
}
