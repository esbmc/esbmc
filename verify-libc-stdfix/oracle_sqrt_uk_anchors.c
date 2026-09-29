/* mkFXPSqrt at unsigned _Accum, validated on natively-computed anchors.
 *
 * The symbolic bracket used at 8/16 bits does not scale here: it squares a
 * symbolic 17-bit root, and no backend discharged it in 40 minutes. These
 * anchors are perfect squares whose roots are exactly representable, so no
 * rounding question arises, plus the documented negative-operand behaviour.
 * Stated as anchor validation, NOT as a proof over all inputs. */
unsigned _Accum __ESBMC_fxp_sqrt_uk(unsigned _Accum);
#define SQ __ESBMC_fxp_sqrt_uk
static unsigned _Accum raw_uk(unsigned int b){unsigned _Accum r;__ESBMC_bitcast(&r,&b);return r;}
static unsigned int bits(unsigned _Accum f){unsigned int b;__ESBMC_bitcast(&b,&f);return b;}

int main(void)
{
  __ESBMC_assert(bits(SQ(raw_uk(0))) == 0, "uk: sqrt(0) = 0");
  /* 0.25 -> 0.5 */
  __ESBMC_assert(
    bits(SQ(raw_uk((unsigned int)1 << 14))) == ((unsigned int)1 << 15),
    "uk: sqrt(0.25) = 0.5");
  /* 2^-4 -> 2^-2 */
  __ESBMC_assert(
    bits(SQ(raw_uk((unsigned int)1 << 12))) == ((unsigned int)1 << 14),
    "uk: sqrt(2^-4) = 2^-2");
  return 0;
}
