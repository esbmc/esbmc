/* Fixed-point division rounds down (floor), NOT toward zero: the exact
 * quotient -86.677/128 floors to -87/128. Asserting the truncated value
 * must fail; a solver that truncates would wrongly verify this. */
#include <assert.h>

int main(void)
{
  short _Fract a = -0.671875hr; /* -86/128 */
  short _Fract b = 0.9921875hr; /* 127/128 */
  assert(a / b == -0.671875hr); /* trunc(-86.677) = -86: WRONG under floor */
  return 0;
}
