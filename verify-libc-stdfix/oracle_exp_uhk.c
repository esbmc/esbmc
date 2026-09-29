/* mkFXPExp at unsigned short _Accum -- one of the six formats camada supports but stdfix.h
 * gives no entry point for, so it is exercised here only against the operation
 * itself.
 *
 * Every expected value below was computed natively in long double and rounded
 * to nearest with ties to even, matching camada's documented contract. None is
 * guessed. */
unsigned short _Accum __ESBMC_fxp_exp_uhk(unsigned short _Accum);
#define EX __ESBMC_fxp_exp_uhk
static unsigned short _Accum raw_uhk(unsigned short b){unsigned short _Accum r;__ESBMC_bitcast(&r,&b);return r;}
static unsigned short bits(unsigned short _Accum f){unsigned short b;__ESBMC_bitcast(&b,&f);return b;}

int main(void)
{
  __ESBMC_assert(
    bits(EX(raw_uhk((unsigned short)0))) == (unsigned short)256,
    "uhk: exp(0/2^8) = raw 256");
  __ESBMC_assert(
    bits(EX(raw_uhk((unsigned short)256))) == (unsigned short)696,
    "uhk: exp(256/2^8) = raw 696");
  __ESBMC_assert(
    bits(EX(raw_uhk((unsigned short)512))) == (unsigned short)1892,
    "uhk: exp(512/2^8) = raw 1892");
  return 0;
}
