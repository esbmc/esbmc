/* One path jumps from outside the loop to its back edge, past its head, with
 * y == 1. Unwound, the false invariant x == 0 cannot hide the assignment at
 * i == 50, so the assertion fails. */
#include <assert.h>
extern int nondet_int(void);
int main()
{
  int i = 0, y = 0, x = 0;
  if (nondet_int())
  {
    y = 1;
    goto end;
  }
  __ESBMC_loop_invariant(i >= 0 && i <= 100 && x == 0);
top:
  i = i + 1;
  if (y == 1 && i == 50)
    x = 1;
  assert(x == 0);
end:
  if (i < 100)
    goto top;
  return 0;
}
