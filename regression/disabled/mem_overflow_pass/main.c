/* mem_overflow_pass:
 * Contract constrains inputs so that a + b cannot overflow a signed 32-bit int.
 * requires(a >= 0 && a <= 1000 && b >= 0 && b <= 1000) keeps sum <= 2000,
 * far below INT_MAX.  --overflow-check finds no violation.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

int bounded_add(int a, int b)
{
  __ESBMC_requires(a >= 0 && a <= 1000);
  __ESBMC_requires(b >= 0 && b <= 1000);
  __ESBMC_ensures(__ESBMC_return_value == a + b);
  __ESBMC_ensures(__ESBMC_return_value >= 0 && __ESBMC_return_value <= 2000);
  return a + b;
}

int main() { return 0; }
