// #7896: _Float16 is IEEE binary16: 10 fraction bits, so the gap above 1 is
// 2^-10 and 1 + 2^-10 is representable. Under the default encoding a wrong
// fraction never reaches this arithmetic: Bitwuzla rejects any 16-bit FP sort
// other than binary16. The github_7896_ulp_fp2bv twin is the test that pins
// the fraction from below by arithmetic, because --fp2bv has no FP sort.
#include <assert.h>

_Float16 nondet_f16(void);

int main(void)
{
  _Float16 x = nondet_f16();
  __ESBMC_assume((float)x == 1.0f);
  _Float16 y = x + (_Float16)0x1p-10f;
  assert(y != x);
  return 0;
}
