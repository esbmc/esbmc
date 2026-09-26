/* Compound shift-assign on fixed-point types. C11 6.5.7p3 promotes a shift's
 * operands independently and takes the result type from the left one, so the
 * right operand is a bit COUNT, not a value in the shifted type: `x >>= 1`
 * must shift by one bit, not by one ulp. Values pinned by native execution. */
#include <assert.h>

int main(void)
{
  short _Fract x = 0.5hr;
  x >>= 1;
  assert(x == 0.25hr);

  short _Fract y = 0.25hr;
  y <<= 1;
  assert(y == 0.5hr);

  unsigned short _Fract u = 0.5uhr;
  u >>= 1;
  assert(u == 0.25uhr);

  /* runtime count */
  int n = 2;
  short _Fract v = 0.5hr;
  v >>= n;
  assert(v == 0.125hr);
  return 0;
}
