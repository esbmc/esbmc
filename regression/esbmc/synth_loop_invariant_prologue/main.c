/* A branch sits in the prologue ahead of the counter and accumulator entry
 * assignments. The invariant-ownership scan walks back from the loop head and
 * must stop at that control flow rather than run on, while the entry-value
 * scan still finds the literals that do reach the head, so the loop is
 * summarised. The bound is unconstrained, so only the synthesised invariant
 * discharges the assertion. */
#include <assert.h>

int main(void)
{
  unsigned int n, c;

  if (c > 100)
    return 0;

  unsigned int pad = 0;
  unsigned int i = 0;
  unsigned int s = 0;

  while (i < n)
  {
    s = s + 1;
    i = i + 1;
  }

  assert(s == n);
  return 0;
}
