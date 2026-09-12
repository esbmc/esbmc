/* A call in the body can write the counter or an accumulator out of sight, so
 * the recogniser declines rather than summarise a body it cannot see. */
#include <assert.h>

static unsigned int step(unsigned int x)
{
  return x + 1;
}

int main(void)
{
  unsigned int i = 0;
  unsigned int s = 0;

  while (i < 4)
  {
    s = s + step(0);
    i = i + 1;
  }

  assert(s == 4);
  return 0;
}
