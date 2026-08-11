// ffsl and ffsll had no model at all, so a call to either got a
// nondeterministic return from symex rather than the bit index it asks for.
// ffs was modelled; the three are checked together here so the family cannot
// drift apart. #183
// Declared rather than pulled from <strings.h>, which Windows does not ship --
// the same reason the model guards its own include.
#include <assert.h>

int ffs(int);
int ffsl(long);
int ffsll(long long);

int main(void)
{
  assert(ffs(0) == 0);
  assert(ffs(1) == 1);
  assert(ffs(8) == 4);
  assert(ffs(-1) == 1);
  // The sign bit is the operand's last, and reaching it must not depend on how
  // a negative value shifts right (C11 6.5.7p5).
  assert(ffs((int)0x80000000u) == 32);

  assert(ffsl(0L) == 0);
  assert(ffsl(1L) == 1);
  assert(ffsl(1L << 20) == 21);
#if __SIZEOF_LONG__ >= 8
  // Only where long is 64-bit: LLP64 (Windows) would shift past its width.
  assert(ffsl(1L << 40) == 41);
#endif
  assert(ffsl(-1L) == 1);

  assert(ffsll(0LL) == 0);
  assert(ffsll(1LL) == 1);
  assert(ffsll(1LL << 60) == 61);
  assert(ffsll(-1LL) == 1);

  return 0;
}
