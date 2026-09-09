/* Mixed-format == and != : TR 18037 4.1.4 inserts no conversions between
 * fixed-point operands, so the comparison sees two different formats and
 * must scale-align them rather than compare raw representations. */
#include <assert.h>

int main(void)
{
  short _Fract a = 0.5hr; /* s.7  raw 0x40 */
  _Accum k = 0.5k;        /* s16.15 raw 0x4000 */
  assert(a == k);

  _Accum k2 = 0.25k;
  assert(a != k2);
  return 0;
}
