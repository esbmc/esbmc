/* Companion to github_4715_irep2_bodies_cond_cov_01_fail: confirms that the
 * --irep2-bodies value-location back-fill (which runs over every round-tripped
 * function body) does not perturb the body's semantics. A short-circuit
 * expression spanning a called function and main still verifies to the same
 * SUCCESSFUL verdict as the flag-off run. */
#include <assert.h>
#include <stdbool.h>

bool both(int a, int b)
{
  return (a > 0) && (b > 0);
}

int main()
{
  int a = 2, b = 3;
  bool r = both(a, b) || (a == b);
  assert(r);
  return 0;
}
