/* mkFXPExp at long _Accum -- one of the six formats camada supports but stdfix.h
 * gives no entry point for, so it is exercised here only against the operation
 * itself.
 *
 * Every expected value below was computed natively in long double and rounded
 * to nearest with ties to even, matching camada's documented contract. None is
 * guessed. */
long _Accum __ESBMC_fxp_exp_lk(long _Accum);
#define EX __ESBMC_fxp_exp_lk
static long _Accum raw_lk(long long b){long _Accum r;__ESBMC_bitcast(&r,&b);return r;}
static long long bits(long _Accum f){long long b;__ESBMC_bitcast(&b,&f);return b;}

int main(void)
{
  __ESBMC_assert(
    bits(EX(raw_lk((long long)0))) == (long long)2147483648,
    "lk: exp(0/2^31) = raw 2147483648");
  __ESBMC_assert(
    bits(EX(raw_lk((long long)2147483648))) == (long long)5837465777,
    "lk: exp(2147483648/2^31) = raw 5837465777");
  __ESBMC_assert(
    bits(EX(raw_lk((long long)-2147483648))) == (long long)790015084,
    "lk: exp(-2147483648/2^31) = raw 790015084");
  __ESBMC_assert(
    bits(EX(raw_lk((long long)4294967296))) == (long long)15867877147,
    "lk: exp(4294967296/2^31) = raw 15867877147");
  return 0;
}
