/* mkFXPExp at unsigned long _Accum -- one of the six formats camada supports but stdfix.h
 * gives no entry point for, so it is exercised here only against the operation
 * itself.
 *
 * Every expected value below was computed natively in long double and rounded
 * to nearest with ties to even, matching camada's documented contract. None is
 * guessed. */
unsigned long _Accum __ESBMC_fxp_exp_ulk(unsigned long _Accum);
#define EX __ESBMC_fxp_exp_ulk
static unsigned long _Accum raw_ulk(unsigned long long b){unsigned long _Accum r;__ESBMC_bitcast(&r,&b);return r;}
static unsigned long long bits(unsigned long _Accum f){unsigned long long b;__ESBMC_bitcast(&b,&f);return b;}

int main(void)
{
  __ESBMC_assert(
    bits(EX(raw_ulk((unsigned long long)0))) == (unsigned long long)4294967296,
    "ulk: exp(0/2^32) = raw 4294967296");
  __ESBMC_assert(
    bits(EX(raw_ulk((unsigned long long)4294967296))) == (unsigned long long)11674931555,
    "ulk: exp(4294967296/2^32) = raw 11674931555");
  __ESBMC_assert(
    bits(EX(raw_ulk((unsigned long long)8589934592))) == (unsigned long long)31735754293,
    "ulk: exp(8589934592/2^32) = raw 31735754293");
  return 0;
}
