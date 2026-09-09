/* All expected values pinned by native execution (clang -ffixed-point). */
#include <assert.h>

int main(void)
{
  _Accum k = -3.5k;
  assert(k + 1.25k == -2.25k);
  assert(k * 2.0k == -7.0k);
  assert(k / 2.0k == -1.75k);
  assert(-k == 3.5k);
  assert(k <= -3.5k);

  unsigned short _Fract u = 0.75uhr;
  assert(u + 0.125uhr == 0.875uhr);
  assert(u - 0.5uhr == 0.25uhr);
  return 0;
}
