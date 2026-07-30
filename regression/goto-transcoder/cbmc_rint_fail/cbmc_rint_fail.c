#include <assert.h>
#include <math.h>
int main()
{
  double a;
  __CPROVER_assume(a == 2.5);
  assert(rint(a) == 3.0); // 2.5 rounds to even (2.0), not 3.0
  return 0;
}
