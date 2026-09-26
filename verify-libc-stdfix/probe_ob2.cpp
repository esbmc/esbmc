#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
extern "C" unsigned short _Accum __ESBMC_fxp_sqrt_uhk(unsigned short _Accum);
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);
int main()
{
  unsigned short n = 65535;
  unsigned short _Accum nv;
  __ESBMC_bitcast(&nv, &n);
  unsigned short _Accum o = __ESBMC_fxp_sqrt_uhk(nv);
  unsigned short ob;
  __ESBMC_bitcast(&ob, &o);
  __ESBMC_assert(ob != 65535, "is it 65535 (MAX)?");
  __ESBMC_assert(ob != 65534, "is it 65534?");
  __ESBMC_assert(ob != 65408, "is it 65408 (=255.5*256)?");
  return 0;
}
