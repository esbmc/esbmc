/* Safe twin of multi_property_falsification. --falsification has no forward
   condition, so it proves nothing; escalating must not turn that into FAILED. */
#include <assert.h>

int main()
{
  unsigned n = nondet_uint();
  unsigned x = 0;
  __ESBMC_assume(n < 5);
  for (unsigned i = 0; i < n; ++i)
    ++x;
  assert(x == n);
  assert(x < 5);
}
