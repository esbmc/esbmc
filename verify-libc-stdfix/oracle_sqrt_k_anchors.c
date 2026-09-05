/* mkFXPSqrt at _Accum, validated on natively-computed anchors.
 *
 * The symbolic bracket used at 8/16 bits does not scale here: it squares a
 * symbolic 16-bit root, and no backend discharged it in 40 minutes. These
 * anchors are perfect squares whose roots are exactly representable, so no
 * rounding question arises, plus the documented negative-operand behaviour.
 * Stated as anchor validation, NOT as a proof over all inputs. */
_Accum __ESBMC_fxp_sqrt_k(_Accum);
#define SQ __ESBMC_fxp_sqrt_k
static _Accum raw_k(int b){_Accum r;__ESBMC_bitcast(&r,&b);return r;}
static int bits(_Accum f){int b;__ESBMC_bitcast(&b,&f);return b;}

int main(void)
{
  __ESBMC_assert(bits(SQ(raw_k(0))) == 0, "k: sqrt(0) = 0");
  /* 0.25 -> 0.5 */
  __ESBMC_assert(
    bits(SQ(raw_k((int)1 << 13))) == ((int)1 << 14),
    "k: sqrt(0.25) = 0.5");
  /* 2^-4 -> 2^-2 */
  __ESBMC_assert(
    bits(SQ(raw_k((int)1 << 11))) == ((int)1 << 13),
    "k: sqrt(2^-4) = 2^-2");

  /* camada documents zero for negative operands (no real square root). */
  __ESBMC_assert(bits(SQ(raw_k(-1))) == 0, "k: sqrt(neg ulp) = 0");
  __ESBMC_assert(
    bits(SQ(raw_k(-((int)1 << 14)))) == 0, "k: sqrt(-0.5) = 0");
  return 0;
}
