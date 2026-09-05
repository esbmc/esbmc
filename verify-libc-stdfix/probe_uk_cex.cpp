// Pin ESBMC to the exact cex input and print what IT computes, to find whether
// the disagreement is in the cex assignment or in the 128-bit arithmetic.
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

  // does ESBMC agree with clang that libc returns raw 3037046839?
  __ESBMC_assert(rb == 3037046839u, "ESBMC agrees libc gives raw 3037046839");

  __uint128_t lo = (__uint128_t)rb * rb;
  __uint128_t hi = ((__uint128_t)rb + 1) * ((__uint128_t)rb + 1);
  __uint128_t ns = (__uint128_t)n << 32;

  __ESBMC_assert(lo <= ns, "lower bracket holds at the cex input");
  __ESBMC_assert(ns < hi, "upper bracket holds at the cex input");
  return 0;
}
