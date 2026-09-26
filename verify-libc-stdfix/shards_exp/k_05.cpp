// expk (s16.15) vs camada's mkFXPExp -- SHARD 5/32.
//
// The 32 shards partition [INT32_MIN, INT32_MAX] exactly: 134217728 inputs
// each, 2^32 total, no gap and no overlap. Verified programmatically.
//
// expk claims a relative bound on one range-reduction step only
// ("relative errors < |lo|^3/2 <= 2^-16"), not end to end, so the property is
// exact agreement with camada's correctly-rounded exp. Where a shard fails, the
// oracle value is the correct answer.
//
// Correct rounding at s16.15 needs a 37-bit intermediate (camada's measured
// hardest-to-round bound), so these are far heavier than the exphk shards.
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
  accum xr = ((x + O) >> (FXRep::FRACTION_LEN - 4)) << (FXRep::FRACTION_LEN - 4);
  accum lo = x - xr;
  S idx = cpp::bit_cast<S>((xr + 0x1.62p3k) >> (FXRep::FRACTION_LEN - 4));
  accum l1 = 0x1.0p0k + (lo >> 1), l2 = 0x1.0p0k + lo * l1;
  return (EXP_HI[idx >> 4] * (EXP_MID[idx & 0xf] * l2));
}

int main()
{
  int xb = nondet_int();
  __ESBMC_assume(xb >= -1476395008 && xb <= -1342177281);   /* shard 5 of 32 */

  _Accum x;
  __ESBMC_bitcast(&x, &xb);

  _Accum l = expk_body(x);
  _Accum o = __ESBMC_fxp_exp_k(x);

  int lb, ob;
  __ESBMC_bitcast(&lb, &l);
  __ESBMC_bitcast(&ob, &o);

  __ESBMC_assert(lb == ob, "expk shard 5: agrees with the exact exp");
  return 0;
}
