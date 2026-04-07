/* ptr_output_param_fail:
 * Body computes the minimum instead of the maximum.
 * The ensures still claims maximum semantics -> VERIFICATION FAILED.
 */
#include <stddef.h>

void compute_max(int a, int b, int *result)
{
  __ESBMC_requires(result != NULL);
  __ESBMC_ensures(*result == (a >= b ? a : b));

  *result = (a <= b) ? a : b; /* wrong: computes min */
}

int main()
{
  int r;
  compute_max(10, 7, &r);
  return 0;
}
