/* A base case discharges a claim only within k, so --falsification, which
   never proves anything, must not report x != 3 as passed: it is violated at
   k = 4. */
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
