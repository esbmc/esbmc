/* Safe twin: the forward condition closes at k = 5 and proves both claims.
   Its own unwinding assertion stays out of the table. */
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
