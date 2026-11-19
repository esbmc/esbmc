#include <assert.h>

int main()
{
  unsigned _BitInt(20) um = 0;
  um = ~um;
  um = um >> 1;
  signed _BitInt(20) m = (signed _BitInt(20))um;
  signed _BitInt(20) r = m - 1 + 1;
  assert(r == m);
  r -= 1;
  r += 1;
  assert(r == m);
  unsigned _BitInt(20) wrap = um + um + 2;
  assert(wrap == 0);
  return 0;
}
