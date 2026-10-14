#include <assert.h>
int main()
{
  unsigned __int128 a, b;
  __CPROVER_assume(a == ((unsigned __int128)1 << 32));
  __CPROVER_assume(b == ((unsigned __int128)1 << 32));
  unsigned __int128 p = a * b;
  assert(p == ((unsigned __int128)1 << 63));   // WRONG: product is 2^64
  return 0;
}
