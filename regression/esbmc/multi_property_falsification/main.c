/* The second claim is violated only at k = 4, so --falsification must keep
   escalating past the first violation to report it (esbmc/esbmc#7900). */
#include <assert.h>

int main()
{
  unsigned n = nondet_uint();
  unsigned x = 0;
  assert(n != 7);
  for (unsigned i = 0; i < n; ++i)
    ++x;
  assert(x != 3);
}
