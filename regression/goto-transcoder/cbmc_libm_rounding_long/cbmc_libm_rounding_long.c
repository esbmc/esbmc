#include <math.h>
int main()
{
  __CPROVER_assert(ceill(2.25L) == 3.0L, "ceill");
  __CPROVER_assert(floorl(2.75L) == 2.0L, "floorl");
  __CPROVER_assert(truncl(-2.75L) == -2.0L, "truncl");
  __CPROVER_assert(roundl(2.5L) == 3.0L, "roundl");
  return 0;
}
