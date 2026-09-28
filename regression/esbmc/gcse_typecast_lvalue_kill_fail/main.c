#include <assert.h>

// Reusing the stale `(long)b + 1` made y equal x (#7992).
int main()
{
  _Bool b = 1;
  long x = (long)b + 1;
  b ^= 1;
  long y = (long)b + 1;
  assert(y == x);
}
