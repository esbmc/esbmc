/* ptr_output_param_pass:
 * Tests a function that uses a pointer purely as an output parameter.
 * The function computes the maximum of two integers and writes it
 * through a pointer.  The ensures clause is a conditional expression
 * over the input arguments — not a simple field assignment.
 */
#include <assert.h>
#include <stddef.h>

void compute_max(int a, int b, int *result)
{
  __ESBMC_requires(result != NULL);
  __ESBMC_ensures(*result == (a >= b ? a : b));

  *result = (a >= b) ? a : b;
}

int main()
{
  int r;
  compute_max(10, 7, &r);
  assert(r == 10);
  return 0;
}
