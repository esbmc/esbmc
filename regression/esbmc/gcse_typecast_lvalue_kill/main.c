#include <assert.h>

// `b ^= 1` writes through `(signed int)b`; that store must also invalidate
// `(long)b + 1`, which does not contain the target expression (#7992).
int main()
{
  _Bool b = 1;
  long x = (long)b + 1;
  b ^= 1;
  long y = (long)b + 1;
  assert(x == 2);
  assert(y == 1);
}
