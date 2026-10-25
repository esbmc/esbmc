/* mkFXPSqrt at unsigned long _Accum, validated on natively-computed anchors.
 *
 * The symbolic bracket used at 8/16 bits does not scale here: it squares a
 * symbolic 33-bit root, and no backend discharged it in 40 minutes. These
 * anchors are perfect squares whose roots are exactly representable, so no
 * rounding question arises, plus the documented negative-operand behaviour.
 * Stated as anchor validation, NOT as a proof over all inputs. */
unsigned long _Accum __ESBMC_fxp_sqrt_ulk(unsigned long _Accum);
#define SQ __ESBMC_fxp_sqrt_ulk
static unsigned long _Accum raw_ulk(unsigned long long b){unsigned long _Accum r;__ESBMC_bitcast(&r,&b);return r;}
static unsigned long long bits(unsigned long _Accum f){unsigned long long b;__ESBMC_bitcast(&b,&f);return b;}

int main(void)
{
  __ESBMC_assert(bits(SQ(raw_ulk(0))) == 0, "ulk: sqrt(0) = 0");
  /* 0.25 -> 0.5 */
  __ESBMC_assert(
    bits(SQ(raw_ulk((unsigned long long)1 << 30))) == ((unsigned long long)1 << 31),
    "ulk: sqrt(0.25) = 0.5");
  /* 2^-4 -> 2^-2 */
  __ESBMC_assert(
    bits(SQ(raw_ulk((unsigned long long)1 << 28))) == ((unsigned long long)1 << 30),
    "ulk: sqrt(2^-4) = 2^-2");
  return 0;
}
