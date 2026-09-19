/* Two reads of one location through a pointer the value-set cannot resolve must
 * agree when nothing is written between them. A scalar read involves no
 * int-to-ptr reconstruction, so the failed symbol's identity is the only thing
 * deciding it: minting a fresh one per read made the two free to differ
 * (#5369). --no-pointer-check keeps the unresolvable dereference itself from
 * being reported, which would mask the assertion. */
#include <assert.h>

unsigned long nondet_ulong(void);

int main(void)
{
  int *p = (int *)nondet_ulong();
  int a = *p;
  int b = *p;
  assert(a == b);
  return 0;
}
