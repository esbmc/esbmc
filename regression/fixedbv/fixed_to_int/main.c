/* Fixed -> integer: truncate toward zero, then C's modular integer
 * conversion. The destination does NOT saturate, even from a _Sat source
 * (saturation belongs to fixed-point destinations). Values pinned by
 * native execution: 4294967293 = -3 mod 2^32, 4464 = 70000 mod 2^16. */
#include <assert.h>

int main(void)
{
  _Accum k = -3.5k;
  assert((int)k == -3);
  assert((unsigned)k == 4294967293u);
  assert((int)3.5k == 3);

  long _Accum big = 70000.25lk;
  assert((unsigned short)big == 4464);
  assert((short)big == 4464);
  return 0;
}
