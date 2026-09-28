#include <assert.h>
#include <limits.h>
int main()
{
  int a, b, c, d, e, f, g, h;
  // INT_MAX + 1 overflows
  __CPROVER_assume(a == INT_MAX && b == 1);
  assert(__builtin_add_overflow_p(a, b, (int)0));
  // INT_MIN - 1 overflows
  __CPROVER_assume(c == INT_MIN && d == 1);
  assert(__builtin_sub_overflow_p(c, d, (int)0));
  // 100000 * 100000 = 1e10 overflows a 32-bit int
  __CPROVER_assume(e == 100000 && f == 100000);
  assert(__builtin_mul_overflow_p(e, f, (int)0));
  // 2 + 3 does not overflow
  __CPROVER_assume(g == 2 && h == 3);
  assert(!__builtin_add_overflow_p(g, h, (int)0));
  return 0;
}
