#include <assert.h>
int main()
{
  unsigned int x;
  __CPROVER_assume(x == 0x12345678u);
  assert(__builtin_rotateleft32(x, 4) == 0x12345678u); // real answer is 0x23456781, must FAIL
  return 0;
}
