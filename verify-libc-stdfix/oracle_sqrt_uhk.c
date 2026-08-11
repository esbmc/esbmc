/* Validate mkFXPSqrt at unsigned short _Accum (u8.8).
 * camada's sqrt is format-generic (exact integer digit recurrence), so every
 * TR 18037 format is in scope -- not only the ones stdfix.h instantiates.
 *
 * The bracket uniquely characterises truncated square root, so proving it
 * proves the operation. Products on raw integers at full width; computing them
 * in the fixed-point type would round and make the bracket meaningless. */
unsigned short _Accum __ESBMC_fxp_sqrt_uhk(unsigned short _Accum);
unsigned short nondet_raw(void);

int main(void)
{
  unsigned short xb = nondet_raw();
  unsigned short _Accum x;
  __ESBMC_bitcast(&x, &xb);
  unsigned short _Accum r = __ESBMC_fxp_sqrt_uhk(x);
  unsigned short rb;
  __ESBMC_bitcast(&rb, &r);

  unsigned long long xs = (unsigned long long)xb << 8;
  unsigned long long lo = (unsigned long long)rb * (unsigned long long)rb;
  unsigned long long hi = ((unsigned long long)rb + 1) * ((unsigned long long)rb + 1);

  __ESBMC_assert(lo <= xs, "uhk: raw_r^2 <= raw_x * 2^8");
  __ESBMC_assert(hi > xs, "uhk: (raw_r+1)^2 > raw_x * 2^8");
  return 0;
}
