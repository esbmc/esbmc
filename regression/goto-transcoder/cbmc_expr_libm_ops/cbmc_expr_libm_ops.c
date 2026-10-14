#include <math.h>
#include <stdlib.h>
int main() {
  __CPROVER_assert(nearbyint(2.5) == 2.0, "nearbyint ties to even");
  __CPROVER_assert(fma(2.0, 3.0, 4.0) == 10.0, "fma is 2*3+4");
  __CPROVER_assert(abs(-7) == 7, "integer abs");
  return 0;
}
