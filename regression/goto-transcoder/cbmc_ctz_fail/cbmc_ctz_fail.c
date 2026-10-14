#include <assert.h>
int main()
{
  unsigned x;
  __CPROVER_assume(x == 0x80000000u);
  assert(__builtin_ctz(x) == 30);
  return 0;
}