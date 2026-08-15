#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
extern "C" unsigned short nondet_ushort();
extern "C" void __ESBMC_bitcast(void *, void *);
namespace fx = LIBC_NAMESPACE::fixed_point;
int main()
{
  unsigned short b = nondet_ushort();
  unsigned short x;
  __ESBMC_bitcast(&x, &b);
  volatile auto r = fx::isqrt_fast(x);
  (void)r;
  return 0;
}
