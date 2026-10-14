#include <math.h>
int main()
{
  __CPROVER_assert(ceilf(2.25f) == 2.0f, "ceilf wrong");
  __CPROVER_assert(floorf(2.75f) == 2.0f, "floorf");
  __CPROVER_assert(truncf(-2.75f) == -2.0f, "truncf");
  __CPROVER_assert(roundf(2.5f) == 3.0f, "roundf");
  __CPROVER_assert(ceil(2.25) == 3.0, "ceil");
  __CPROVER_assert(floor(2.75) == 2.0, "floor");
  __CPROVER_assert(trunc(-2.75) == -2.0, "trunc");
  __CPROVER_assert(round(2.5) == 3.0, "round");
  return 0;
}
