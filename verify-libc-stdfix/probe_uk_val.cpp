#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);
namespace fx = LIBC_NAMESPACE::fixed_point;
int main()
{
  unsigned int n = 2147549183u;
  unsigned _Accum r = fx::isqrt(n);
  unsigned int rb;
  __ESBMC_bitcast(&rb, &r);
  /* bisect ESBMC's value */
  __ESBMC_assert(rb > 3037046839u, "ESBMC value > clang's 3037046839?");
  return 0;
}
