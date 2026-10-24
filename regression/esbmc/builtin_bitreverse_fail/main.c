#include <assert.h>

int main()
{
  assert(__builtin_bitreverse16(0x1234) == 0x2c48);
  assert(__builtin_bitreverse16(0x1234) == 0x4321);
  return 0;
}
