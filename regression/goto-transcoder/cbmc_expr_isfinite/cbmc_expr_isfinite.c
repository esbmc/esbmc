#include <math.h>
int main() {
  double d = 1.5;
  double inf = HUGE_VAL;
  __CPROVER_assert(__CPROVER_isfinited(d), "1.5 is finite");
  __CPROVER_assert(!__CPROVER_isfinited(inf), "HUGE_VAL is not finite");
  return 0;
}
