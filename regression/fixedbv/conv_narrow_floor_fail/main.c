/* Narrowing floors: -38.40234375/128 floors to -39/128. Asserting the
 * toward-zero value (-38/128) must fail; an encoder that truncates
 * narrowing conversions would wrongly verify this. */
#include <assert.h>

int main(void)
{
  _Fract w = -0.300048828125r; /* s.15 -9831/32768 */
  assert((short _Fract)w == -0.296875hr); /* trunc -38/128: WRONG */
  return 0;
}
