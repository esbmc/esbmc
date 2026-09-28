#include <assert.h>
int main()
{
  unsigned __int128 a, b;
  __CPROVER_assume(a == ((unsigned __int128)1 << 32));   // 2^32
  __CPROVER_assume(b == ((unsigned __int128)1 << 32));   // 2^32
  unsigned __int128 p = a * b;                            // 2^64, fits in 128 bits
  assert(p == ((unsigned __int128)1 << 64));
  return 0;
}
