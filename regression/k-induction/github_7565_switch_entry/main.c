/* The switch enters the loop at three points, none of them the head's
 * fall-through predecessor, so each entering jump needs its own havoc (#7565).
 * The trip count keeps the forward condition out of reach, so the inductive
 * step is what closes it: every path through the body ends at `s = 3`, which
 * holds from the arbitrary state each entry havoc leaves behind. */

#include <assert.h>

extern int nondet_int(void);

int main()
{
  int n = nondet_int();
  int i = 0, s = 0;
  if (n < 0 || n > 3)
    return 0;
  switch (n)
  {
  case 0:
    goto l0;
  case 1:
    goto l1;
  default:
    goto l2;
  }
l0:
  s = 1;
l1:
  s = 2;
l2:
  s = 3;
  i = i + 1;
  assert(s == 3);
  if (i < 100)
    goto l0;
  return 0;
}
