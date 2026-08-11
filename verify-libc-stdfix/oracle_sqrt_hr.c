/* Validate mkFXPSqrt at short _Fract (s1.7).
 * camada's sqrt is format-generic (exact integer digit recurrence), so every
 * TR 18037 format is in scope -- not only the ones stdfix.h instantiates.
 *
 * The bracket uniquely characterises truncated square root, so proving it
 * proves the operation. Products on raw integers at full width; computing them
 * in the fixed-point type would round and make the bracket meaningless. */
short _Fract __ESBMC_fxp_sqrt_hr(short _Fract);
signed char nondet_raw(void);

int main(void)
{
  signed char xb = nondet_raw();
  short _Fract x;
  __ESBMC_bitcast(&x, &xb);
  short _Fract r = __ESBMC_fxp_sqrt_hr(x);
  signed char rb;
  __ESBMC_bitcast(&rb, &r);

  /* Negative operands have no real square root; camada documents zero. */
  if (xb < 0)
  {
    __ESBMC_assert(rb == 0, "sqrt of a negative hr value is zero");
    return 0;
  }

  int xs = (int)xb << 7;
  int lo = (int)rb * (int)rb;
  int hi = ((int)rb + 1) * ((int)rb + 1);

  __ESBMC_assert(lo <= xs, "hr: raw_r^2 <= raw_x * 2^7");
  __ESBMC_assert(hi > xs, "hr: (raw_r+1)^2 > raw_x * 2^7");
  return 0;
}
