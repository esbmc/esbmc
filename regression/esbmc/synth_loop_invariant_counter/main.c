/* Counter-only loop: the synthesised bound (i < n || i == n || i == 0) is
 * enough on its own; no accumulator conjunct is needed. */
#include <assert.h>

int main(void)
{
  unsigned int n;
  unsigned int i = 0;

  while (i < n)
    i++;

  assert(i == n || n == 0);
  return 0;
}
