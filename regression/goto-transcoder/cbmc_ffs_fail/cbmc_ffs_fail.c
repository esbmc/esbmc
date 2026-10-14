#include <assert.h>
int main()
{
  int x;
  __CPROVER_assume(x == 0x100);   // lowest set bit is bit 8 -> ffs is 9
  assert(__builtin_ffs(x) == 8);  // off by one: must FAIL, proving it is computed
  return 0;
}
