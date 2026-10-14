/* A write between two reads of an unresolvable location must be observed: a
 * fix for #5369 that shares the free value keyed on the pointer alone misses
 * it, because a write through an unresolved pointer leaves the pointer's own
 * SSA value unchanged. */
#include <assert.h>

unsigned long nondet_ulong(void);

int main(void)
{
  int *p = (int *)nondet_ulong();
  int a = *p;
  *p = a + 1;
  int b = *p;
  assert(a == b);
  return 0;
}
