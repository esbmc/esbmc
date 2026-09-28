#include <math.h>
int main() {
  double d = 4.0;
  __CPROVER_assert(sqrt(d) == 2.0, "sqrt(4)==2");
  return 0;
}
