#include <assert.h>

unsigned nondet_uint(void);

int main()
{
  assert(__builtin_rotateleft32(0x80000001u, 1) == 3);
  assert(__builtin_rotateright8(1, 1) == 0x80);
  assert(__builtin_rotateleft16(0x1234, 4) == 0x2341);
  assert(__builtin_rotateright64(1, 65) == 1ull << 63);

  unsigned x = nondet_uint(), n = nondet_uint();
  assert(__builtin_rotateleft32(x, 32) == x);
  assert(__builtin_rotateright32(__builtin_rotateleft32(x, n), n) == x);
  return 0;
}
