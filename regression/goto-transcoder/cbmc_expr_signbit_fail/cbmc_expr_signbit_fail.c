#include <math.h>
int main() {
  double d = -1.5, e = 1.5;
  __CPROVER_assert(signbit(d), "-1.5 has the sign bit set");
  __CPROVER_assert(signbit(e), "1.5 wrongly claimed negative");
  return 0;
}
