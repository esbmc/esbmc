/* Validate mkFXPSqrt at unsigned _Accum (u16.16).
 * camada's sqrt is format-generic (exact integer digit recurrence), so every
 * TR 18037 format is in scope -- not only the ones stdfix.h instantiates.
 *
 * The bracket uniquely characterises truncated square root, so proving it
 * proves the operation. Products on raw integers at full width; computing them
 * in the fixed-point type would round and make the bracket meaningless. */
unsigned _Accum __ESBMC_fxp_sqrt_uk(unsigned _Accum);
unsigned int nondet_raw(void);

int main(void)
{
  unsigned int xb = nondet_raw();
  unsigned _Accum x;
  __ESBMC_bitcast(&x, &xb);
  unsigned _Accum r = __ESBMC_fxp_sqrt_uk(x);
  unsigned int rb;
  __ESBMC_bitcast(&rb, &r);

  __uint128_t xs = (__uint128_t)xb << 16;
  __uint128_t lo = (__uint128_t)rb * (__uint128_t)rb;
  __uint128_t hi = ((__uint128_t)rb + 1) * ((__uint128_t)rb + 1);

  __ESBMC_assert(lo <= xs, "uk: raw_r^2 <= raw_x * 2^16");
  __ESBMC_assert(hi > xs, "uk: (raw_r+1)^2 > raw_x * 2^16");
  return 0;
}
