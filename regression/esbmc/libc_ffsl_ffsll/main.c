// ffsl and ffsll had no model at all, so a call to either got a
// nondeterministic return from symex rather than the bit index it asks for.
// ffs was modelled; the three are checked together here so the family cannot
// drift apart. #183
#include <assert.h>
#include <strings.h>

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
  assert(ffsl(1L << 40) == 41);
  assert(ffsl(-1L) == 1);

  assert(ffsll(0LL) == 0);
  assert(ffsll(1LL) == 1);
  assert(ffsll(1LL << 60) == 61);
  assert(ffsll(-1LL) == 1);

  return 0;
}
