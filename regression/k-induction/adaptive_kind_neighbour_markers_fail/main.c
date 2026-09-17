/* The loop entered past its head must not be cut, but the schema attached
 * the markers emitted for the loop before it and cut it anyway, missing the
 * bug at i == 50 (#7565). */
#include <assert.h>
extern int nondet_int(void);
int main()
{
  int n = nondet_int();
  int m = nondet_int();
  int i = 0, s = 0, x = 0, t = 0;
  if (n < 0 || n > 1)
    return 0;
  while (t < 3)
    t++;
  if (n == 0)
    goto l1;
  goto l2;
l0:
  s = s + 1;
l1:
  s = s + 2;
l2:
  i = i + 1;
  if (i == 50)
    x = 1;
  assert(x == 0);
  if (i < m)
    goto l0;
  return 0;
}
