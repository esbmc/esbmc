#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/fixed_point/fx_bits.h"
extern "C" short nondet_short();
extern "C" void __ESBMC_bitcast(void *, void *);
namespace fx = LIBC_NAMESPACE::fixed_point;
int main()
{
  short b = nondet_short();
  short _Accum x;
  __ESBMC_bitcast(&x, &b);
  volatile auto r = fx::countls(x);
  (void)r;
  return 0;
}
