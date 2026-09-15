/* The counterpart to github_5369_scalar: two *different* unresolvable locations
 * must still be free to hold different values, so sharing is keyed on the
 * location rather than applied to every unresolved read alike. */
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
