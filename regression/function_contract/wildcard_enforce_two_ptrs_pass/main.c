#include <stddef.h>

/* Regression test: --enforce-contract '*' --function add_ptrs with TWO pointer
 * parameters.  Verifies that add_pointer_validity_assumptions allocates backing
 * storage for every pointer parameter, not just the first one.  Without the
 * fix both *a and *b would be unallocated, causing alignment faults. */

int add_ptrs(const int *a, const int *b)
{
  __ESBMC_requires(a != NULL);
  __ESBMC_requires(b != NULL);
  __ESBMC_assigns();
  __ESBMC_ensures(__ESBMC_return_value == *a + *b);
  return *a + *b;
}

int main()
{
  int x = 3, y = 4;
  int res = add_ptrs(&x, &y);
  return 0;
}
