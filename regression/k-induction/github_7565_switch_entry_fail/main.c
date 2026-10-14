/* The switch dispatches through unconditional trampolines that enter the loop
 * past its head, so the head's havoc is unreachable and, before #7565, nothing
 * havoced the loop. The inductive step then ran from the concrete entry state
 * and proved the program at k = 2, missing the bug at i == 50. */

#include <assert.h>

extern int nondet_int(void);

int main()
{
  int n = nondet_int();
  int i = 0, s = 0, x = 0;
  if (n < 0 || n > 1)
    return 0;
  switch (n)
  {
  case 0:
    goto l1;
  default:
    goto l2;
  }
l0:
  s = s + 1;
l1:
  s = s + 2;
l2:
  i = i + 1;
  if (i == 50)
    x = 1;
  assert(x == 0);
  if (i < 100)
    goto l0;
  return 0;
}
