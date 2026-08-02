#include <assert.h>
int main()
{
  unsigned long long x;
  __CPROVER_assume(x == 0x100ULL);
  assert(__builtin_ctzll(x) == 8);
  return 0;
}