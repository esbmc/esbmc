// C11 7.12.3.5: isnormal is false for zero, subnormal, infinite and NaN (#7320).
#include <assert.h>
#include <float.h>
#include <math.h>

int main(void)
{
  double inf = 1.0 / 0.0;

  assert(!__builtin_isnormal(inf));
  assert(!__builtin_isnormal(-inf));
  assert(!__builtin_isnormal(DBL_MAX * 2.0));
  assert(!__builtin_isnormal(-DBL_MAX * 2.0));
  assert(!__builtin_isnormal(0.0 / 1.0));
  assert(!__builtin_isnormal(DBL_MIN / 2.0));
  assert(!__builtin_isnormal(0.0 / 0.0));

  assert(__builtin_isnormal(1.0));
  assert(__builtin_isnormal(DBL_MAX));
  assert(__builtin_isnormal(DBL_MIN));

  return 0;
}
