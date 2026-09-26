/* Fixed -> fixed narrowing rounds down (floor), matching Clang; pinned by
 * native execution. Contrast with fixed -> int, which truncates toward
 * zero. */
#include <assert.h>

int main(void)
{
  _Fract w = -0.300048828125r; /* s.15 raw 0xd999 = -9831/32768 */
  /* *128 = -38.40234375 floors to -39: s.7 -39/128 */
  assert((short _Fract)w == -0.3046875hr);

  _Fract wp = 0.300048828125r; /* positive tail: floor == trunc */
  assert((short _Fract)wp == 0.296875hr); /* 38/128 */

  assert((short _Fract)0.25r == 0.25hr); /* exact stays exact */

  /* boundary: -0.99609375 * 128 = -127.5 floors to -128, the s.7 minimum
   * itself (one step further would be out of range = UB) */
  _Accum ka = -0.99609375k;
  assert((short _Fract)ka == -0.9921875hr - 0.0078125hr);
  return 0;
}
