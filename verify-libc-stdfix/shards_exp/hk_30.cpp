// exphk (s8.7) vs camada's mkFXPExp -- SHARD 30/32.
//
// The 32 shards partition [INT16_MIN, INT16_MAX] exactly: 2048 inputs each,
// 65536 total, no gap and no overlap (verified programmatically). If every
// shard verifies, exphk is checked against the exact operation over its whole
// domain.
//
// mkFXPExp is exp correctly rounded to nearest with ties to even, saturating at
// the format maximum and flushing below half an ulp. exphk documents only a
// relative bound on one range-reduction step ("relative errors < |lo|^2 <=
// 2^-8"), not an end-to-end claim -- so the property asserted here is exact
// agreement with the oracle, which is what a caller of a function named exp
// would infer. Where it fails, the oracle value is the correct answer.
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/CPP/bit.h"
#include "hdr/stdint_proxy.h"

extern "C" short _Accum __ESBMC_fxp_exp_hk(short _Accum);
extern "C" short nondet_short();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);
extern "C" void __ESBMC_bitcast(void *, void *);

namespace fx = LIBC_NAMESPACE::fixed_point;
namespace cpp = LIBC_NAMESPACE::cpp;

static constexpr short accum EXP_HI[12] = {
  0x1.0p-7hk, 0x1.0p-6hk, 0x1.8p-5hk,  0x1.1p-3hk,  0x1.78p-2hk,  0x1.0p0hk,
  0x1.5cp1hk, 0x1.d9p2hk, 0x1.416p4hk, 0x1.b4dp5hk, 0x1.28d4p7hk, SACCUM_MAX,
};
static constexpr short accum EXP_MID[8] = {
  0x1.38p-1hk, 0x1.6p-1hk, 0x1.9p-1hk, 0x1.c4p-1hk,
  0x1.0p0hk,   0x1.22p0hk, 0x1.48p0hk, 0x1.74p0hk,
};
static short accum exphk_body(short accum x)
{
  using FXRep = fx::FXRep<short accum>;
  using StorageType = typename FXRep::StorageType;
  if (x >= 0x1.64p2hk)
    return FXRep::MAX();
  if (x <= -0x1.63p2hk)
    return FXRep::ZERO();
  constexpr short accum ONE_SIXTEENTH = 0x1.0p-4hk;
  short accum x_rounded =
    ((x + ONE_SIXTEENTH) >> (FXRep::FRACTION_LEN - 3))
    << (FXRep::FRACTION_LEN - 3);
  short accum lo = x - x_rounded;
  StorageType indices = cpp::bit_cast<StorageType>(
    (x_rounded + 0x1.6p2hk) >> (FXRep::FRACTION_LEN - 3));
  short accum exp_hi = EXP_HI[indices >> 3];
  short accum exp_mid = EXP_MID[indices & 0x7];
  return (exp_hi * (exp_mid * (0x1.0p0hk + lo)));
}

int main()
{
  short xb = nondet_short();
  __ESBMC_assume(xb >= 28672 && xb <= 30719);   /* shard 30 of 32 */

  short _Accum x;
  __ESBMC_bitcast(&x, &xb);

  short _Accum l = exphk_body(x);
  short _Accum o = __ESBMC_fxp_exp_hk(x);

  short lb, ob;
  __ESBMC_bitcast(&lb, &l);
  __ESBMC_bitcast(&ob, &o);

  __ESBMC_assert(lb == ob, "exphk shard 30: agrees with the exact exp");
  return 0;
}
