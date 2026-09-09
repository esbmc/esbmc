/* mkFXPExp at unsigned _Accum -- one of the six formats camada supports but stdfix.h
 * gives no entry point for, so it is exercised here only against the operation
 * itself.
 *
 * Every expected value below was computed natively in long double and rounded
 * to nearest with ties to even, matching camada's documented contract. None is
 * guessed. */
unsigned _Accum __ESBMC_fxp_exp_uk(unsigned _Accum);
#define EX __ESBMC_fxp_exp_uk
static unsigned _Accum raw_uk(unsigned int b){unsigned _Accum r;__ESBMC_bitcast(&r,&b);return r;}
static unsigned int bits(unsigned _Accum f){unsigned int b;__ESBMC_bitcast(&b,&f);return b;}

int main(void)
{
  __ESBMC_assert(
    bits(EX(raw_uk((unsigned int)0))) == (unsigned int)65536,
    "uk: exp(0/2^16) = raw 65536");
  __ESBMC_assert(
    bits(EX(raw_uk((unsigned int)65536))) == (unsigned int)178145,
    "uk: exp(65536/2^16) = raw 178145");
  __ESBMC_assert(
    bits(EX(raw_uk((unsigned int)131072))) == (unsigned int)484249,
    "uk: exp(131072/2^16) = raw 484249");
  return 0;
}
