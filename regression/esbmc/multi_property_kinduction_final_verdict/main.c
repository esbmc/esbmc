/* esbmc/esbmc#7900 D1: the loop cannot be exhausted, so no phase settles the
   program and the run reaches --max-k-step with a violation already reported.
   The verdict must follow the counterexample, not the exhausted bound. */
#include <assert.h>
int main()
{
  int x = 0;
  while (1)
  {
    assert(x < 3);
    assert(x != 10);
    ++x;
  }
}
