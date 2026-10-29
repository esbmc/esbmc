#include <assert.h>
#include <stdlib.h>

/* A _BitInt(256) aligns to the target's _BitInt cap (8 on x86-64, 16 on
 * AArch64), not to its 32-byte width, so both reads below are aligned. */
typedef unsigned _BitInt(256) u256;

static u256 rd(const u256 *p)
{
  return *p;
}

int main(void)
{
  u256 x = 5;
  assert(rd(&x) == 5);

  unsigned char *m = calloc(64, 1);
  if (!m)
    return 0;
  assert(rd((const u256 *)(m + 16)) == 0);
  free(m);
  return 0;
}
