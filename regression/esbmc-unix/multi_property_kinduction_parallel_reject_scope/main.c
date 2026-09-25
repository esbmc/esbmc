/* esbmc/esbmc#7900: the parallel driver cannot merge per-claim verdicts. */
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
