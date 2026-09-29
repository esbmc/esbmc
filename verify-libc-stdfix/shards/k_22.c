/* Validate mkFXPSqrt at _Accum (s16.15) -- SHARD 22/32.
 * camada's sqrt is format-generic (exact integer digit recurrence), so every
 * TR 18037 format is in scope -- not only the ones stdfix.h instantiates.
 *
 * The bracket uniquely characterises truncated square root, so proving it
 * proves the operation. Products on raw integers at full width; computing them
 * in the fixed-point type would round and make the bracket meaningless. */
_Accum __ESBMC_fxp_sqrt_k(_Accum);
int nondet_raw(void);

int main(void)
{
  int xb = nondet_raw();
  /* Shard 22 of 32: a disjoint slice of the int32 domain. The 32 shards
   * partition [INT32_MIN, INT32_MAX] exactly -- no gaps, no overlaps -- so if
   * every shard verifies SUCCESSFUL the property holds over the whole domain.
   * This is a complete case split, not sampling. */
  __ESBMC_assume(xb >= 805306368 && xb <= 939524095);
  _Accum x;
  __ESBMC_bitcast(&x, &xb);
  _Accum r = __ESBMC_fxp_sqrt_k(x);
  int rb;
  __ESBMC_bitcast(&rb, &r);

  /* Negative operands have no real square root; camada documents zero. */
  if (xb < 0)
  {
    __ESBMC_assert(rb == 0, "sqrt of a negative k value is zero");
    return 0;
  }

  __int128 xs = (__int128)xb << 15;
  __int128 lo = (__int128)rb * (__int128)rb;
  __int128 hi = ((__int128)rb + 1) * ((__int128)rb + 1);

  __ESBMC_assert(lo <= xs, "k: raw_r^2 <= raw_x * 2^15");
  __ESBMC_assert(hi > xs, "k: (raw_r+1)^2 > raw_x * 2^15");
  return 0;
}
