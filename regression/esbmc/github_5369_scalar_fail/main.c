/* Two *different* unresolvable locations must stay free to hold different
 * values: a fix for #5369 that shares one free value across every unresolved
 * read alike, rather than keying on the location, turns this SUCCESSFUL. */
#include <assert.h>

unsigned long nondet_ulong(void);

int main(void)
{
  int *p = (int *)nondet_ulong();
  int *q = (int *)nondet_ulong();
  int a = *p;
  int b = *q;
  assert(a == b);
  return 0;
}
