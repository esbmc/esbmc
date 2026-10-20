#include <assert.h>

int main()
{
  unsigned _BitInt(40) um = 0;
  um = ~um;
  um = um >> 1;
  signed _BitInt(40) m = (signed _BitInt(40))um;
  signed _BitInt(40) r = m - 1 + 1;
  assert(r == m);
  unsigned _BitInt(40) wrap = um + um + 2;
  assert(wrap == 0);
  return 0;
}
