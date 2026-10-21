/* The -1.0hr literal trap: per TR 18037 4.1.5 the 1.0 fract literal is the
 * MAXIMUM, so -1.0hr = -127/128 (raw 0x81) while the min clamp produces
 * -128/128 (raw 0x80). This assertion must fail; an encoder that reads
 * -1.0hr as the minimum would wrongly verify it. */
#include <assert.h>

int main(void)
{
  _Sat short _Fract s = -0.75hr;
  s = s - 0.75hr; /* clamps to min: raw 0x80 */
  assert(s == -1.0hr); /* -1.0hr is raw 0x81: WRONG */
  return 0;
}
