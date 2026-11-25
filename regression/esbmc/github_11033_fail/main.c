#include <stdbool.h>

/* Issue #11033: a branching invariant after another one hid the first. */
bool f_true()
{
  return true;
}

int main()
{
  int idx0 = 0;
  int idx1 = 0;
  __ESBMC_loop_invariant(idx0 == idx1 + 1);
  __ESBMC_loop_invariant(f_true() || f_true());
  while (idx0 < 10)
  {
    __ESBMC_assert(idx0 == idx1, "top: equal");
    ++idx0;
    idx1 = idx0;
  }
  return 0;
}
