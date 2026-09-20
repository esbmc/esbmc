/* esbmc/esbmc#7900 D1: the trip count is nondeterministic, so the forward
   condition never closes and the run reaches --max-k-step having already
   reported two violations. The verdict must follow the counterexamples. */
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
