#include <assert.h>
int main()
{
  int x, y;
  unsigned long long z;
  __CPROVER_assume(x == 0x100);           // lowest set bit is bit 8
  __CPROVER_assume(y == 0);               // zero input
  __CPROVER_assume(z == 0x100000000ULL);  // lowest set bit is bit 32
  assert(__builtin_ffs(x) == 9);          // 1-based index of the lowest set bit
  assert(__builtin_ffs(y) == 0);          // zero-input guard (load-bearing case)
  assert(__builtin_ffsll(z) == 33);       // width-generality (64-bit operand)
  return 0;
}
