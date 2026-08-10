// How far below the exact root does libc go? Tighten the bound until it holds.
// The oracle truncates, so the exact root is in [rb, rb+1); "libc >= rb - k"
// failing for k means the error exceeds k ulp downward.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
extern "C" unsigned short _Fract __ESBMC_fxp_sqrt_uhr(unsigned short _Fract);
extern "C" unsigned char nondet_uchar();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_bitcast(void *, void *);
namespace fx = LIBC_NAMESPACE::fixed_point;
int main()
{
  unsigned char xb = nondet_uchar();
  unsigned short _Fract x;
  __ESBMC_bitcast(&x, &xb);
  unsigned short _Fract lr = fx::sqrt(x), rr = __ESBMC_fxp_sqrt_uhr(x);
  unsigned char lb, rb;
  __ESBMC_bitcast(&lb, &lr);
  __ESBMC_bitcast(&rb, &rr);
  __ESBMC_assert(
    (int)lb >= (int)rb - 1, "libc sqrt is at most 1 ulp BELOW the exact root");
  return 0;
}
