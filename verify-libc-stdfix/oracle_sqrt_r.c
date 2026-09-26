/* Validate mkFXPSqrt at _Fract (s1.15).
 * camada's sqrt is format-generic (exact integer digit recurrence), so every
 * TR 18037 format is in scope -- not only the ones stdfix.h instantiates.
 *
 * The bracket uniquely characterises truncated square root, so proving it
 * proves the operation. Products on raw integers at full width; computing them
 * in the fixed-point type would round and make the bracket meaningless. */
_Fract __ESBMC_fxp_sqrt_r(_Fract);
short nondet_raw(void);

int main(void)
{
  short xb = nondet_raw();
  _Fract x;
  __ESBMC_bitcast(&x, &xb);
  _Fract r = __ESBMC_fxp_sqrt_r(x);
  short rb;
  __ESBMC_bitcast(&rb, &r);

  /* Negative operands have no real square root; camada documents zero. */
  if (xb < 0)
  {
    __ESBMC_assert(rb == 0, "sqrt of a negative r value is zero");
    return 0;
  }

  long long xs = (long long)xb << 15;
  long long lo = (long long)rb * (long long)rb;
  long long hi = ((long long)rb + 1) * ((long long)rb + 1);

  __ESBMC_assert(lo <= xs, "r: raw_r^2 <= raw_x * 2^15");
  __ESBMC_assert(hi > xs, "r: (raw_r+1)^2 > raw_x * 2^15");
  return 0;
}
