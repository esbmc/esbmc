// expk (s16.15) against camada's mkFXPExp.
//
// Two results, and the difference between them matters:
//
//   * Defect A of exphk does NOT recur here. expk's EXP_HI[23] is also an
//     ACCUM_MAX placeholder (exp(12) = 162754.79 does not fit s16.15), but it
//     is UNREACHABLE: the maximum index reached is 22, matching libc's own
//     "indices <= 355" comment (355 >> 4 = 22). So exphk's reachable
//     placeholder is a bug specific to exphk's narrower table, not a shared
//     design flaw. Worth stating explicitly -- it would have been easy to
//     assume the pattern generalised.
//
//   * Defect B DOES recur. EXP_HI[0] is 0x1p-15, exactly one ulp, and `lo` can
//     be negative, so exp_hi * (exp_mid * l2) underflows to zero just inside
//     the domain. The guard flushes only x <= -11.0903320, while
//
//         exp(-11.0898132) = 0.0000152671   half an ulp = 0.0000152588
//
//     is above half an ulp, so the correctly-rounded result is raw 1, not 0.
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/CPP/bit.h"
#include "hdr/stdint_proxy.h"

extern "C" _Accum __ESBMC_fxp_exp_k(_Accum);
extern "C" int nondet_int();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;
namespace cpp = LIBC_NAMESPACE::cpp;

static constexpr accum EXP_HI[24] = {
    0x1p-15k,0x1p-15k,0x1p-13k,0x1.6p-12k,0x1.ep-11k,0x1.44p-9k,0x1.bap-8k,
    0x1.2cp-6k,0x1.97cp-5k,0x1.153p-3k,0x1.78b8p-2k,0x1p0k,0x1.5bf1p1k,
    0x1.d8e68p2k,0x1.415e6p4k,0x1.b4c9p5k,0x1.28d388p7k,0x1.936dc6p8k,
    0x1.1228858p10k,0x1.749ea7cp11k,0x1.fa7157cp12k,0x1.5829dcf8p14k,
    0x1.d3c4489p15k,ACCUM_MAX,};
static constexpr accum EXP_MID[16] = {
    0x1.e0fcp-1k,0x1p0k,0x1.1082p0k,0x1.2216p0k,0x1.34ccp0k,0x1.48b6p0k,
    0x1.5deap0k,0x1.747ap0k,0x1.8c8p0k,0x1.a612p0k,0x1.c14cp0k,0x1.de46p0k,
    0x1.fd1ep0k,0x1.0efap1k,0x1.2074p1k,0x1.330ep1k,};
static accum expk_body(accum x)
{
  using FXRep = fx::FXRep<accum>;
  using S = typename FXRep::StorageType;
  if (x >= 0x1.62e4p3k)
    return FXRep::MAX();
  if (x <= -0x1.62e44p3k)
    return FXRep::ZERO();
  constexpr accum O = 0x1.0p-5k;
  accum xr = ((x + O) >> (FXRep::FRACTION_LEN - 4))
             << (FXRep::FRACTION_LEN - 4);
  accum lo = x - xr;
  S idx = cpp::bit_cast<S>((xr + 0x1.62p3k) >> (FXRep::FRACTION_LEN - 4));
  accum l1 = 0x1.0p0k + (lo >> 1), l2 = 0x1.0p0k + lo * l1;
  return (EXP_HI[idx >> 4] * (EXP_MID[idx & 0xf] * l2));
}

int main()
{
  /* Symbolic over the whole flush window, not a handful of pinned inputs. */
  int xb = nondet_int();
  __ESBMC_assume(xb >= -363391 && xb <= -363360);

  _Accum x;
  __ESBMC_bitcast(&x, &xb);
  _Accum lr = expk_body(x);
  _Accum rr = __ESBMC_fxp_exp_k(x);

  int lb, rb;
  __ESBMC_bitcast(&lb, &lr);
  __ESBMC_bitcast(&rb, &rr);

  __ESBMC_assume(rb != 0);
  __ESBMC_assert(lb != 0, "expk does not flush a representable value to zero");
  return 0;
}
