/* The body writes through an index (`w[i] = 1`). That is not a self-increment
 * of a symbol, and it may alias the counter, so the recogniser declines. */
#include <assert.h>

int main(void)
{
  unsigned int w[4];
  unsigned int i = 0;

  while (i < 4)
  {
    w[i] = 1;
    i = i + 1;
  }

  assert(w[0] == 1);
  return 0;
}
