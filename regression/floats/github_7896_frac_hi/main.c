/* github #7896: binary16 has 10 significand bits, so the gap above 1 is
 * 2^-10 and 1 + 2^-11 rounds back to 1. Under --floatbv a wrong significand
 * usually makes Bitwuzla reject the sort instead, so this copy is the erroring
 * control; github_7896_frac_hi_fp2bv is the arithmetic one. */
#include <assert.h>

_Float16 nondet_h(void);

int main(void)
{
  _Float16 x = nondet_h();
  __ESBMC_assume((float)x == 1.0f);
  _Float16 y = x + (_Float16)0x1p-11f;
  assert(y != x);
  return 0;
}
