#include <assert.h>
int main()
{
  unsigned a, b, res;
  __CPROVER_assume(a == 100000u && b == 100000u);
  int of = __builtin_mul_overflow(a, b, &res);
  assert(!of);   // WRONG: 1e10 overflows 32-bit unsigned
  return 0;
}
