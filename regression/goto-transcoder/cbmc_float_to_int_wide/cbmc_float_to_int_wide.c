#include <stdint.h>
int main()
{
  __float128 q = 3.5Q;
  long double ld = 2.5L;
  int64_t a = (int64_t)q;
  int32_t b = (int32_t)ld;
  __CPROVER_assert(a == 3, "float128 -> i64");
  __CPROVER_assert(b == 2, "long double -> i32");
  return 0;
}
