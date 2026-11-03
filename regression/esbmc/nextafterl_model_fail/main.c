#include <assert.h>
#include <float.h>
#include <math.h>

int main(void)
{
  assert(nextafterl(1.0L, 2.0L) == 1.0L);
  assert(nextafterl(1.0L, 0.0L) == 1.0L - LDBL_EPSILON);
  assert(nexttoward(1.0, 0.0L) == 1.0 - DBL_EPSILON);
  assert(nexttowardf(1.0f, 2.0L) == 1.0f);
  return 0;
}
