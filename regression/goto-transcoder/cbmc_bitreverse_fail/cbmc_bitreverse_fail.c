#include <assert.h>
int main()
{
  unsigned int x;
  __CPROVER_assume(x == 0x1u);
  assert(__builtin_bitreverse32(x) == 0x1u); // real answer is 0x80000000, must FAIL
  return 0;
}
