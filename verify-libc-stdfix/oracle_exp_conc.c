/* Oracle against natively-established ground truth (correctly rounded to
 * nearest, ties to even, in s8.7). Values measured, not guessed:
 *   raw    0 ->   128   exp(0)=1
 *   raw  128 ->   348   exp(1)=2.71828 -> 2.71875
 *   raw -128 ->    47   exp(-1)=0.36788 -> 0.3671875
 *   raw  256 ->   946   exp(2)=7.38906 -> 7.390625
 *   raw  640 -> 18997   exp(5)=148.41316 -> 148.4140625
 *   raw  704 -> 31321   exp(5.5)=244.69193 -> 244.6953125
 *   raw  710 -> 32767   exp(5.54688)=256.43 -> saturates to MAX
 *   raw -800 ->     0   exp(-6.25)=0.00193 -> below half an ulp -> zero */
short _Accum __ESBMC_fxp_exp_hk(short _Accum);
static short _Accum raw(short b){short _Accum r;__ESBMC_bitcast(&r,&b);return r;}
static short bits(short _Accum f){short b;__ESBMC_bitcast(&b,&f);return b;}
int main(void)
{
  __ESBMC_assert(bits(__ESBMC_fxp_exp_hk(raw(   0))) ==   128, "exp(0)");
  __ESBMC_assert(bits(__ESBMC_fxp_exp_hk(raw( 128))) ==   348, "exp(1)");
  __ESBMC_assert(bits(__ESBMC_fxp_exp_hk(raw(-128))) ==    47, "exp(-1)");
  __ESBMC_assert(bits(__ESBMC_fxp_exp_hk(raw( 256))) ==   946, "exp(2)");
  __ESBMC_assert(bits(__ESBMC_fxp_exp_hk(raw( 640))) == 18997, "exp(5)");
  __ESBMC_assert(bits(__ESBMC_fxp_exp_hk(raw( 704))) == 31321, "exp(5.5)");
  __ESBMC_assert(bits(__ESBMC_fxp_exp_hk(raw( 710))) == 32767, "exp saturates");
  __ESBMC_assert(bits(__ESBMC_fxp_exp_hk(raw(-800))) ==     0, "exp flushes to 0");
  return 0;
}
