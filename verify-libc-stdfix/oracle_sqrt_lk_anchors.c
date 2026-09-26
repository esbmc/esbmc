/* mkFXPSqrt at long _Accum, validated on natively-computed anchors.
 *
 * The symbolic bracket used at 8/16 bits does not scale here: it squares a
 * symbolic 32-bit root, and no backend discharged it in 40 minutes. These
 * anchors are perfect squares whose roots are exactly representable, so no
 * rounding question arises, plus the documented negative-operand behaviour.
 * Stated as anchor validation, NOT as a proof over all inputs. */
long _Accum __ESBMC_fxp_sqrt_lk(long _Accum);
#define SQ __ESBMC_fxp_sqrt_lk
static long _Accum raw_lk(long long b){long _Accum r;__ESBMC_bitcast(&r,&b);return r;}
static long long bits(long _Accum f){long long b;__ESBMC_bitcast(&b,&f);return b;}

int main(void)
{
  __ESBMC_assert(bits(SQ(raw_lk(0))) == 0, "lk: sqrt(0) = 0");
  /* 0.25 -> 0.5 */
  __ESBMC_assert(
    bits(SQ(raw_lk((long long)1 << 29))) == ((long long)1 << 30),
    "lk: sqrt(0.25) = 0.5");
  /* 2^-4 -> 2^-2 */
  __ESBMC_assert(
    bits(SQ(raw_lk((long long)1 << 27))) == ((long long)1 << 29),
    "lk: sqrt(2^-4) = 2^-2");

  /* camada documents zero for negative operands (no real square root). */
  __ESBMC_assert(bits(SQ(raw_lk(-1))) == 0, "lk: sqrt(neg ulp) = 0");
  __ESBMC_assert(
    bits(SQ(raw_lk(-((long long)1 << 30)))) == 0, "lk: sqrt(-0.5) = 0");
  return 0;
}
