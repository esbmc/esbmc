/* The body assignment `s = t + u` is an addition, but not of s to something:
 * neither operand is the target. It is not a self-increment, so the recogniser
 * declines. */
#include <assert.h>

int main(void)
{
  unsigned int t, u;
  __ESBMC_assume(t <= 2 && u <= 2);

  unsigned int i = 0;
  unsigned int s = 0;

  while (i < 4)
  {
    s = t + u;
    i = i + 1;
  }

  assert(s <= 4);
  return 0;
}
