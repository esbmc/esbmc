#include <assert.h>
int main()
{
  long long x = 1;
  __int128 y = (__int128)1 << 100;
  /* 6.3.1.8: the long long operand promotes to __int128 before the add. */
  assert(x + y == ((__int128)1 << 100) + 1);
  return 0;
}
