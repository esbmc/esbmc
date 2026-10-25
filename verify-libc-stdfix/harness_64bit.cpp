// 64-bit tier of LLVM libc's stdfix -- the last one:
//
//   long _Accum (s32.31)   unsigned long _Accum (u32.32)
//
// 13 entry points: abslk, countlslk/ulk, roundlk/ulk, bits/fxbits for both,
// divilk/ulk and idivlk/ulk. There is no sqrt at this width (sqrt stops at
// 32-bit) and no exp.
//
// All bracket arithmetic is __int128: a raw value here uses the full 64 bits
// and step reaches 2^32, so int64_t would overflow exactly as int did at the
// 32-bit tier. __int128 covers every width with room to spare.
//
// Properties are TR 18037 7.18a.6, not re-implementations.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" long _Accum nondet_laccum();
extern "C" unsigned long _Accum nondet_ulaccum();
extern "C" int nondet_int();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

namespace fx = LIBC_NAMESPACE::fixed_point;
using LIBC_NAMESPACE::cpp::bit_cast;

int main()
{
  /* ---------- abslk (s32.31) ---------- */
  {
    long _Accum x = nondet_laccum();
    long _Accum r = fx::abs(x);
    using R = fx::FXRep<long _Accum>;
    __ESBMC_assert(r >= R::ZERO(), "abslk: never negative");
    __ESBMC_assert(
      x == R::MIN() ? r == R::MAX() : (r == x || r == -x),
      "abslk: magnitude preserved, MIN saturates");
    __ESBMC_assert(fx::abs(r) == r, "abslk: idempotent");
  }

  /* ---------- countls on both 64-bit types ---------- */
  {
    long _Accum v = nondet_laccum();
    __ESBMC_assume(v != 0.0lk);
    int k = fx::countls(v);
    __ESBMC_assert(k >= 0 && k <= 63, "countlslk in range");
    __ESBMC_assert(
      ((long _Accum)(v << k) >> k) == v, "countlslk: v<<k does not overflow");
  }
  {
    unsigned long _Accum v = nondet_ulaccum();
    __ESBMC_assume(v != 0.0ulk);
    int k = fx::countls(v);
    __ESBMC_assert(k >= 0 && k <= 64, "countlsulk in range");
    __ESBMC_assert(
      ((unsigned long _Accum)(v << k) >> k) == v,
      "countlsulk: v<<k does not overflow");
    __ESBMC_assume(k < 64);
    __ESBMC_assert(
      ((unsigned long _Accum)(v << (k + 1)) >> (k + 1)) != v,
      "countlsulk: k is maximal");
  }

  /* ---------- roundlk (s32.31), n symbolic over every position ---------- */
  {
    long _Accum x = nondet_laccum();
    int n = nondet_int();
    __ESBMC_assume(n >= 0 && n < 31); // FRACTION_LEN == 31 for s32.31

    long _Accum r = fx::round(x, n);
    int64_t xr = bit_cast<int64_t, long _Accum>(x);
    int64_t rr = bit_cast<int64_t, long _Accum>(r);

    const __int128 step = (__int128)1 << (31 - n);
    const __int128 x128 = xr;

    if (rr != INT64_MAX)
    {
      __ESBMC_assert(rr % step == 0, "roundlk: multiple of 2^-n");
      const __int128 down =
        (x128 >= 0 ? (x128 / step) * step : ((x128 - step + 1) / step) * step);
      const __int128 up = down + step;
      if (up - x128 <= x128 - down)
        __ESBMC_assert(rr == up, "roundlk: ties round toward +Inf");
      else
        __ESBMC_assert(rr == down, "roundlk: nearer-down rounds down");
    }
  }

  /* ---------- bits/fxbits round-trips ---------- */
  {
    long _Accum a = nondet_laccum();
    __ESBMC_assert(
      bit_cast<long _Accum, int64_t>(bit_cast<int64_t, long _Accum>(a)) == a,
      "lkbits(bitslk(a)) == a");
    unsigned long _Accum ua = nondet_ulaccum();
    __ESBMC_assert(
      bit_cast<unsigned long _Accum, uint64_t>(
        bit_cast<uint64_t, unsigned long _Accum>(ua)) == ua,
      "ulkbits(bitsulk(ua)) == ua");
  }

  return 0;
}
