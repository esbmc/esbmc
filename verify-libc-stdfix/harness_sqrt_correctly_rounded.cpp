// Is fixed_point::sqrt CORRECTLY ROUNDED? That is a stronger question than any
// bound: does it return the nearest representable value to the true root for
// every input?
//
// camada's mkFXPSqrt truncates toward zero, so the true root lies in
// [oracle, oracle+1ulp). The nearest representable value is therefore either
// `oracle` or `oracle+1`, decided by which half of that interval the root falls
// in -- comparable without computing the root, by squaring the midpoint:
//
//     root >= oracle + 1/2   <=>   (2*oracle + 1)^2 <= 4 * raw_x * 2^F
//
// so the correctly rounded result is oracle+1 when that holds, oracle otherwise.
// All integer arithmetic at full width; nothing is re-implemented.
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

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

  unsigned short _Fract l = fx::sqrt(x);
  unsigned short _Fract o = __ESBMC_fxp_sqrt_uhr(x);

  unsigned char lb, ob;
  __ESBMC_bitcast(&lb, &l);
  __ESBMC_bitcast(&ob, &o);

  /* which of {ob, ob+1} is nearest the true root? */
  unsigned long long mid = 2ull * ob + 1ull;
  unsigned long long want = (mid * mid <= 4ull * ((unsigned long long)xb << 8))
                              ? (unsigned long long)ob + 1ull
                              : (unsigned long long)ob;

  __ESBMC_assert(lb == want, "sqrt is correctly rounded (nearest representable)");
  return 0;
}
