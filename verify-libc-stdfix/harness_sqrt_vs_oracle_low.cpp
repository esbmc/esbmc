// LLVM libc's fixed_point::sqrt against camada's mkFXPSqrt, inside the solver.
//
// This is the methodology the earlier sqrt numbers LACKED. Those came from
// running libc natively under clang -ffixed-point and comparing against
// long double -- a differential test over sampled or enumerated inputs. Here
// the reference is an SMT term, so the comparison is a PROOF over all inputs
// of the format, and the reference is not something this harness computes.
//
// camada's mkFXPSqrt is documented as square root rounded TOWARD ZERO: the
// unique r with r*r <= x < (r+1ulp)^2. It is NOT round-to-nearest. So the
// property below is stated against truncation, and the 1-ulp allowance is
// exactly the gap between truncation and nearest -- not slack invented to make
// libc pass.
//
// LLVM libc claims (sqrt.h:211-212) "Absolute errors < 2^(-fraction length)",
// i.e. strictly less than one ulp. Against a truncating reference that means
// libc must be within 1 ulp ABOVE the oracle, and never below it.
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

  unsigned short _Fract libc_r = fx::sqrt(x);   // the real library body
  unsigned short _Fract ref_r = __ESBMC_fxp_sqrt_uhr(x); // the solver's exact op

  unsigned char lb, rb;
  __ESBMC_bitcast(&lb, &libc_r);
  __ESBMC_bitcast(&rb, &ref_r);

  // The oracle truncates, so the exact root lies in [rb, rb+1). libc's
  // documented "absolute error < 1 ulp" therefore permits exactly {rb, rb+1}:
  // anything below rb is more than an ulp under, anything above rb+1 more than
  // an ulp over.
  //
  // Proved separately so one failure cannot mask the other.
  __ESBMC_assert(lb >= rb, "libc sqrt is never below the truncated exact root");
  return 0;
}
