#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
extern "C" unsigned char nondet_uchar();
extern "C" void __ESBMC_bitcast(void *, void *);
namespace fx = LIBC_NAMESPACE::fixed_point;
int main()
{
  unsigned char b = nondet_uchar();
  unsigned short _Fract x;
  __ESBMC_bitcast(&x, &b);
  volatile auto r = fx::sqrt(x);
  (void)r;
  return 0;
}
