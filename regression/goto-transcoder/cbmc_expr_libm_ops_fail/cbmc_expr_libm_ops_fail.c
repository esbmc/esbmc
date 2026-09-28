#include <math.h>
#include <stdlib.h>
int main() {
  __CPROVER_assert(nearbyint(2.5) == 2.0, "nearbyint ties to even");
  __CPROVER_assert(fma(2.0, 3.0, 4.0) == 11.0, "fma wrong");
  __CPROVER_assert(abs(-7) == 7, "integer abs");
  return 0;
}
