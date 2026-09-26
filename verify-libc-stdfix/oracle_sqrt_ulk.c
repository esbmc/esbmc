/* Validate mkFXPSqrt at unsigned long _Accum (u32.32).
 * camada's sqrt is format-generic (exact integer digit recurrence), so every
 * TR 18037 format is in scope -- not only the ones stdfix.h instantiates.
 *
 * The bracket uniquely characterises truncated square root, so proving it
 * proves the operation. Products on raw integers at full width; computing them
 * in the fixed-point type would round and make the bracket meaningless. */
unsigned long _Accum __ESBMC_fxp_sqrt_ulk(unsigned long _Accum);
unsigned long long nondet_raw(void);

int main(void)
{
  unsigned long long xb = nondet_raw();
  unsigned long _Accum x;
  __ESBMC_bitcast(&x, &xb);
  unsigned long _Accum r = __ESBMC_fxp_sqrt_ulk(x);
  unsigned long long rb;
  __ESBMC_bitcast(&rb, &r);

  __uint128_t xs = (__uint128_t)xb << 32;
  __uint128_t lo = (__uint128_t)rb * (__uint128_t)rb;
  __uint128_t hi = ((__uint128_t)rb + 1) * ((__uint128_t)rb + 1);

  __ESBMC_assert(lo <= xs, "ulk: raw_r^2 <= raw_x * 2^32");
  __ESBMC_assert(hi > xs, "ulk: (raw_r+1)^2 > raw_x * 2^32");
  return 0;
}
