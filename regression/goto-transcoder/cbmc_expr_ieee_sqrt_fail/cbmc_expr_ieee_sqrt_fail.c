#include <math.h>
int main() {
  double d = 4.0;
  __CPROVER_assert(sqrt(d) == 3.0, "sqrt(4)==3");
  return 0;
}
