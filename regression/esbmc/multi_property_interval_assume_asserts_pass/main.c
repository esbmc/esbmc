/* The twin of multi_property_interval_assume_asserts: both assertions hold,
   so the run must still end VERIFICATION SUCCESSFUL (esbmc/esbmc#7900). */
#include <assert.h>
int main()
{
  int x = nondet_int();
  __ESBMC_assume(x > 10);
  assert(x > 10);
  assert(x > 5);
  return 0;
}
