#include <stdlib.h>

/* Offset 4 is below the alignment a _BitInt(256) needs on any target. */
typedef unsigned _BitInt(256) u256;

int main(void)
{
  unsigned char *m = calloc(64, 1);
  if (!m)
    return 0;
  u256 *p = (u256 *)(m + 4);
  int r = *p == 0;
  free(m);
  return r;
}
