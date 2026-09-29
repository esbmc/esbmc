/* Integer -> fixed, pinned by native execution. Out-of-range into a plain
 * fixed type is UB, so only in-range and _Sat cases are asserted. */
#include <assert.h>

int main(void)
{
  int i3 = 3;
  assert((_Accum)i3 == 3.0k);
  int im = -2;
  assert((_Accum)im == -2.0k);

  unsigned int ub = 200u;
  assert((_Sat short _Accum)ub == 200.0hk); /* fits, no clamp */
  unsigned int ub2 = 300u;
  assert((_Sat short _Accum)ub2 == 255.9921875hk); /* clamps to s8.7 max */
  assert((_Sat short _Fract)i3 == 0.9921875hr);    /* clamps to fract max */
  assert((_Sat _Fract)1 == 0.999969482421875r);    /* s.15 max */
  return 0;
}
