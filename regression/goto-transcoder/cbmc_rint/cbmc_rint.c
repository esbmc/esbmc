#include <assert.h>
#include <math.h>
int main()
{
  double a;
  __CPROVER_assume(a == 2.5);
  assert(rint(a) == 2.0); // round half to even: 2.5 -> 2

  double b;
  __CPROVER_assume(b == 3.5);
  assert(rint(b) == 4.0); // round half to even: 3.5 -> 4

  float f;
  __CPROVER_assume(f == 2.7f);
  assert(rintf(f) == 3.0f);

  long double l;
  __CPROVER_assume(l == 5.5L);
  assert(rintl(l) == 6.0L); // long double width
  return 0;
}
