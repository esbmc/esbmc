#include <stdbool.h>

/* Issue #10997: the short-circuit branch of the invariant was re-run
 * unconditionally in the base case. */
bool f_true(int arg)
{
  return true;
}

bool f_false(int arg)
{
  return false;
}

int main()
{
  int var1, var2;
  int idx;
  __ESBMC_loop_invariant(f_true(var1) || f_false(var2));
  for (idx = 0; idx < 2; ++idx)
  {
  }
  return 0;
}
