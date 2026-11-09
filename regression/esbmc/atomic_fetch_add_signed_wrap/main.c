#include <limits.h>
#include <stdatomic.h>

atomic_int a = INT_MAX;
int b = INT_MAX;
int c = INT_MIN;
signed char s = SCHAR_MAX;
int d = -1;

int main()
{
  atomic_fetch_add(&a, 1);
  __atomic_fetch_add(&b, 1, __ATOMIC_SEQ_CST);
  __sync_fetch_and_sub(&c, 1);
  signed char t = __atomic_add_fetch(&s, 1, __ATOMIC_SEQ_CST);
  __atomic_fetch_add(&d, 1, __ATOMIC_SEQ_CST);
  __ESBMC_assert(a == INT_MIN && b == INT_MIN, "atomic add wraps");
  __ESBMC_assert(c == INT_MAX, "atomic sub wraps");
  __ESBMC_assert(s == SCHAR_MIN && t == SCHAR_MIN, "narrow add wraps");
  __ESBMC_assert(d == 0, "no unsigned wrap");
  return 0;
}
