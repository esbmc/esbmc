/* Validate mkFXPSqrt at u0.32 -- the widest sqrt format libc instantiates.
 * Same unique characterisation as u0.8/u0.16: truncated square root is the
 * only r satisfying both brackets. Products on raw integers at full width
 * (128-bit here, since raw_x * 2^32 needs 64 bits and r^2 needs 64). */
unsigned long _Fract __ESBMC_fxp_sqrt_ulr(unsigned long _Fract);
unsigned int nondet_uint(void);

int main(void)
{
  unsigned int xb = nondet_uint();
  unsigned long _Fract x;
  __ESBMC_bitcast(&x, &xb);
  unsigned long _Fract r = __ESBMC_fxp_sqrt_ulr(x);
  unsigned int rb;
  __ESBMC_bitcast(&rb, &r);

  __uint128_t rr = (__uint128_t)rb * rb;
  __uint128_t nn = ((__uint128_t)rb + 1) * ((__uint128_t)rb + 1);
  __uint128_t xs = (__uint128_t)xb << 32;

  __ESBMC_assert(rr <= xs, "u0.32: raw_r^2 <= raw_x * 2^32");
  __ESBMC_assert(nn > xs, "u0.32: (raw_r+1)^2 > raw_x * 2^32");
  return 0;
}
