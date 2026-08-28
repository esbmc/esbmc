/* github.com/esbmc/esbmc/issues/6443
 *
 * Pins the is_false guard on the assert-to-assume conversion. One loop, and
 * the claim simplifies to the literal false, so neither multiple_loops_seen
 * nor loop_iterations excludes it: the guard is the only thing keeping this
 * assertion a proof obligation. Converting it yields assume(false), which
 * kills the path, leaves every later claim vacuous and lets the step report
 * the program proved.
 *
 * The inductive step is run on its own because k-induction as a whole is
 * decided by the base case here: a reachable assert(0) is refuted at k = 1,
 * which masks whatever the step does. */
#include <assert.h>
int main(void)
{
  while (1)
  {
    assert(0);
  }

  return 0;
}
