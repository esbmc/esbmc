/* mkFXPSqrt at unsigned long _Fract, validated on natively-computed anchors.
 *
 * The symbolic bracket used at 8/16 bits does not scale here: it squares a
 * symbolic 33-bit root, and no backend discharged it in 40 minutes. These
 * anchors are perfect squares whose roots are exactly representable, so no
 * rounding question arises, plus the documented negative-operand behaviour.
 * Stated as anchor validation, NOT as a proof over all inputs. */
unsigned long _Fract __ESBMC_fxp_sqrt_ulr(unsigned long _Fract);
#define SQ __ESBMC_fxp_sqrt_ulr
static unsigned long _Fract raw_ulr(unsigned int b){unsigned long _Fract r;__ESBMC_bitcast(&r,&b);return r;}
static unsigned int bits(unsigned long _Fract f){unsigned int b;__ESBMC_bitcast(&b,&f);return b;}

int main(void)
{
  __ESBMC_assert(bits(SQ(raw_ulr(0))) == 0, "ulr: sqrt(0) = 0");
  /* 0.25 -> 0.5 */
  __ESBMC_assert(
    bits(SQ(raw_ulr((unsigned int)1 << 30))) == ((unsigned int)1 << 31),
    "ulr: sqrt(0.25) = 0.5");
  /* 2^-4 -> 2^-2 */
  __ESBMC_assert(
    bits(SQ(raw_ulr((unsigned int)1 << 28))) == ((unsigned int)1 << 30),
    "ulr: sqrt(2^-4) = 2^-2");
  return 0;
}
