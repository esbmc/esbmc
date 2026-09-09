/* _Sat clamping at both rails, pinned by native execution.
 * TR 18037 4.1.5: a 1.0 fract literal denotes the MAXIMUM (127/128 for
 * s.7), so -1.0hr is -127/128 (raw 0x81), NOT the minimum -1.0 (raw 0x80);
 * the true minimum is only reachable by arithmetic. */
#include <assert.h>

int main(void)
{
  _Sat short _Fract s = 0.75hr;
  s = s + 0.75hr; /* 1.5 clamps to max = 127/128 */
  assert(s == 0.9921875hr);

  s = -0.75hr;
  s = s - 0.75hr; /* -1.5 clamps to min = -1.0 (raw 0x80) */
  assert(s == -0.9921875hr - 0.0078125hr);
  assert(s + 0.5hr == -0.5hr);

  _Sat unsigned short _Fract su = 0.25uhr;
  su = su - 0.75uhr; /* negative clamps to 0 */
  assert(su == 0.0uhr);

  _Sat short _Accum sk = 200.0hk;
  sk = sk + 200.0hk; /* 400 > max clamps */
  assert(sk == 255.9921875hk); /* s8.7 max = 32767/128 */
  return 0;
}
