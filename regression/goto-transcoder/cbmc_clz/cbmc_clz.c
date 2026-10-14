#include <assert.h>
int main()
{
  unsigned x;
  __CPROVER_assume(x == 0x0000FFFFu);
  assert(__builtin_clz(x) == 16);
  return 0;
}