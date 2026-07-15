#include <assert.h>
int main()
{
  unsigned x;
  __CPROVER_assume(x == 0x12345678u);
  assert(__builtin_bswap32(x) == 0x78563412u);
  return 0;
}
