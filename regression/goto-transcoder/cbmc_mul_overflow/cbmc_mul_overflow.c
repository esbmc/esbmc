#include <assert.h>
int main()
{
  unsigned a, b, res;
  // 100000 * 100000 = 1e10 overflows a 32-bit unsigned;
  // wrapped low 32 bits = 10000000000 mod 2^32 = 1410065408
  __CPROVER_assume(a == 100000u && b == 100000u);
  int of = __builtin_mul_overflow(a, b, &res);
  assert(of);
  assert(res == 1410065408u);

  // 3 * 4 = 12 fits: no overflow, exact value
  unsigned e, f, r3;
  __CPROVER_assume(e == 3u && f == 4u);
  int of3 = __builtin_mul_overflow(e, f, &r3);
  assert(!of3 && r3 == 12u);
  return 0;
}
