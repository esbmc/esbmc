#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
extern "C" unsigned short _Accum __ESBMC_fxp_sqrt_uhk(unsigned short _Accum);
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);
namespace fx = LIBC_NAMESPACE::fixed_point;
int main()
{
  unsigned short n = 65535;
  unsigned short _Accum nv;
  __ESBMC_bitcast(&nv, &n);
  unsigned short _Accum o = __ESBMC_fxp_sqrt_uhk(nv);
  unsigned short ob;
  __ESBMC_bitcast(&ob, &o);
  /* what IS the oracle raw here? bisect */
  __ESBMC_assert(ob != 65280, "ob == 65280?");
  __ESBMC_assert(ob < 65000, "ob < 65000?");
  return 0;
}
