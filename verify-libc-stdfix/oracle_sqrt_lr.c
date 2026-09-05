/* Validate mkFXPSqrt at long _Fract (s1.31).
 * camada's sqrt is format-generic (exact integer digit recurrence), so every
 * TR 18037 format is in scope -- not only the ones stdfix.h instantiates.
 *
 * The bracket uniquely characterises truncated square root, so proving it
 * proves the operation. Products on raw integers at full width; computing them
 * in the fixed-point type would round and make the bracket meaningless. */
long _Fract __ESBMC_fxp_sqrt_lr(long _Fract);
int nondet_raw(void);

int main(void)
{
  int xb = nondet_raw();
  long _Fract x;
  __ESBMC_bitcast(&x, &xb);
  long _Fract r = __ESBMC_fxp_sqrt_lr(x);
  int rb;
  __ESBMC_bitcast(&rb, &r);

  /* Negative operands have no real square root; camada documents zero. */
  if (xb < 0)
  {
    __ESBMC_assert(rb == 0, "sqrt of a negative lr value is zero");
    return 0;
  }

  __int128 xs = (__int128)xb << 31;
  __int128 lo = (__int128)rb * (__int128)rb;
  __int128 hi = ((__int128)rb + 1) * ((__int128)rb + 1);

  __ESBMC_assert(lo <= xs, "lr: raw_r^2 <= raw_x * 2^31");
  __ESBMC_assert(hi > xs, "lr: (raw_r+1)^2 > raw_x * 2^31");
  return 0;
}
