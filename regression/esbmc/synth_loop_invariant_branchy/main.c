/* The body branches, so the per-iteration effect is not affine and the
 * recogniser declines. Declining must stay silent and must not disturb the
 * verdict: the loop is bounded, so BMC still proves the property. */
#include <assert.h>

int main(void)
{
  unsigned int i = 0;
  unsigned int s = 0;

  while (i < 6)
  {
    if (i % 2 == 0)
      s = s + 1;
    else
      s = s + 2;
    i++;
  }

  assert(s == 9);
  return 0;
}
