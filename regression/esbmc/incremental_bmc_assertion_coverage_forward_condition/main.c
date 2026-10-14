/* Both assertions hold and the forward condition closes at k = 5, so the
   coverage report printed on that exit must show both assertion instances
   reached. */
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
