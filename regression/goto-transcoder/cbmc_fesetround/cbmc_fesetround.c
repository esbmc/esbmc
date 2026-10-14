#include <fenv.h>
int main()
{
  double a = 1.0, b = 3.0;
  fesetround(FE_DOWNWARD);
  double lo = a / b;
  fesetround(FE_UPWARD);
  double hi = a / b;
  __CPROVER_assert(lo < hi, "fesetround changes the result of 1/3");
  return 0;
}
