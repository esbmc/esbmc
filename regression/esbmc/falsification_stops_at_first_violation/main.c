/* Without --multi-property, --falsification stops at the first violation,
   so the k = 4 violation is never reached (esbmc/esbmc#7900). */
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
