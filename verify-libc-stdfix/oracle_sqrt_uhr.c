/* Validate mkFXPSqrt at unsigned short _Fract (u0.8).
 * camada's sqrt is format-generic (exact integer digit recurrence), so every
 * TR 18037 format is in scope -- not only the ones stdfix.h instantiates.
 *
 * The bracket uniquely characterises truncated square root, so proving it
 * proves the operation. Products on raw integers at full width; computing them
 * in the fixed-point type would round and make the bracket meaningless. */
unsigned short _Fract __ESBMC_fxp_sqrt_uhr(unsigned short _Fract);
unsigned char nondet_raw(void);

int main(void)
{
  unsigned char xb = nondet_raw();
  unsigned short _Fract x;
  __ESBMC_bitcast(&x, &xb);
  unsigned short _Fract r = __ESBMC_fxp_sqrt_uhr(x);
  unsigned char rb;
  __ESBMC_bitcast(&rb, &r);

  unsigned int xs = (unsigned int)xb << 8;
  unsigned int lo = (unsigned int)rb * (unsigned int)rb;
  unsigned int hi = ((unsigned int)rb + 1) * ((unsigned int)rb + 1);

  __ESBMC_assert(lo <= xs, "uhr: raw_r^2 <= raw_x * 2^8");
  __ESBMC_assert(hi > xs, "uhr: (raw_r+1)^2 > raw_x * 2^8");
  return 0;
}
