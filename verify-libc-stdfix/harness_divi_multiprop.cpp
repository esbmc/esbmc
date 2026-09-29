// Every UB and overflow claim in libc's divi at once, via --multi-property.
// No assumptions narrowing n or d beyond what the source itself calls UB
// (d == 0), and no assertion of my own -- the checkers are the oracle.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" int nondet_int();
extern "C" void __ESBMC_assume(bool);

namespace fx = LIBC_NAMESPACE::fixed_point;

int main()
{
  int n = nondet_int();
  int d = nondet_int();
  __ESBMC_assume(d != 0);   /* source documents divide-by-zero as UB */

  volatile long _Fract r = fx::divi<long _Fract>(n, d);
  (void)r;
  return 0;
}
