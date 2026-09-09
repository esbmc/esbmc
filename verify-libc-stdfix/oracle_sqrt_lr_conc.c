/* Wide-format oracle validation, concretely. The symbolic bracket does not
 * scale past 16 bits -- it squares a symbolic 32/64-bit root -- so validate
 * mkFXPSqrt at these widths on natively-measured anchors instead, and say so.
 * s0.31: raw values whose exact roots are known. */
long _Fract __ESBMC_fxp_sqrt_lr(long _Fract);
static long _Fract raw(int b){long _Fract r;__ESBMC_bitcast(&r,&b);return r;}
static int bits(long _Fract f){int b;__ESBMC_bitcast(&b,&f);return b;}
int main(void)
{
  /* perfect squares in s0.31: (k/2^31)^2 has raw k^2/2^31, so pick k = 2^m */
  __ESBMC_assert(bits(__ESBMC_fxp_sqrt_lr(raw(0))) == 0, "sqrt(0)=0");
  /* 0.25 -> 0.5 : raw 2^29 -> raw 2^30 */
  __ESBMC_assert(
    bits(__ESBMC_fxp_sqrt_lr(raw(1 << 29))) == (1 << 30), "sqrt(0.25)=0.5");
  /* negative operands give zero */
  __ESBMC_assert(bits(__ESBMC_fxp_sqrt_lr(raw(-1))) == 0, "sqrt(neg)=0");
  __ESBMC_assert(
    bits(__ESBMC_fxp_sqrt_lr(raw(-(1 << 30)))) == 0, "sqrt(-0.5)=0");
  return 0;
}
