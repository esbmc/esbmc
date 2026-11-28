/* One path jumps past the loop head with y == 1, bypassing the base case and
 * the havoc there. Unwound, the false invariant x == 0 cannot hide the
 * assignment at i == 50, so the assertion fails. */
#include <assert.h>
extern int nondet_int(void);
int main()
{
  int i = 0, y = 0, x = 0;
  if (nondet_int())
  {
    y = 1;
    goto mid;
  }
  __ESBMC_loop_invariant(i >= 0 && i <= 100 && x == 0);
top:
  i = i + 1;
mid:
  if (y == 1 && i == 50)
    x = 1;
  assert(x == 0);
  if (i < 100)
    goto top;
  return 0;
}
