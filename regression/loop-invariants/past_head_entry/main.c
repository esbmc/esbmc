/* A loop entered past its head is left to the unwinder, so a true invariant
 * still verifies, with a warning that only its base case was checked. */
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
  assert(x == 0);
  if (i < 100)
    goto top;
  return 0;
}
