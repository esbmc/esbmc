#include <assert.h>
int main()
{
  unsigned int x;
  unsigned long long y;
  __CPROVER_assume(x == 0x12345678u);
  __CPROVER_assume(y == 0x8000000000000001ULL);
  assert(__builtin_rotateleft32(x, 4) == 0x23456781u);
  assert(__builtin_rotateright32(x, 4) == 0x81234567u);
  assert(__builtin_rotateleft32(x, 32) == 0x12345678u); // mod width: full rotate = identity
  assert(__builtin_rotateleft64(y, 1) == 0x3ULL);       // 64-bit width
  return 0;
}
