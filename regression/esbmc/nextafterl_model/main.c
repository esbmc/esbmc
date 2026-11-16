#include <assert.h>
#include <float.h>
#include <math.h>

int main(void)
{
  assert(nextafterl(1.0L, 2.0L) == 1.0L + LDBL_EPSILON);
  assert(nextafterl(1.0L, 0.0L) == 1.0L - LDBL_EPSILON / 2);
  assert(nextafterl(-1.0L, -2.0L) == -1.0L - LDBL_EPSILON);
  assert(nextafterl(0.0L, -1.0L) == -LDBL_TRUE_MIN);
  assert(nextafterl(LDBL_MIN, 0.0L) == LDBL_MIN - LDBL_TRUE_MIN);
  assert(nextafterl(LDBL_TRUE_MIN, 0.0L) == 0.0L);
  assert(nextafterl(-INFINITY, 0.0L) == -LDBL_MAX);
  assert(nextafterl(3.0L, 3.0L) == 3.0L);
  assert(isnan(nextafterl(NAN, 1.0L)));
  assert(isinf(nextafterl(LDBL_MAX, INFINITY)));
  assert(nexttoward(1.0, 2.0L) == 1.0 + DBL_EPSILON);
  assert(nexttowardf(1.0f, 0.0L) == 1.0f - FLT_EPSILON / 2);
  assert(nexttowardf(2.0f, 2.0L) == 2.0f);
  assert(nexttowardl(1.0L, 2.0L) == 1.0L + LDBL_EPSILON);
  return 0;
}
