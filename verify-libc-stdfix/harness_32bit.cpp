// 32-bit tier of LLVM libc's stdfix: the four types with 16-bit storage,
//
//   long _Fract (s0.31)   unsigned long _Fract (u0.32)
//   _Accum (s16.15)       unsigned _Accum (u16.16)
//
// covering abs, countls, round and the bits/fxbits pair. The _Accum types are
// the first with INTEGER bits, so the round step, the saturation rail and
// countls's sign accounting all differ from the fract cases already done.
//
// Properties are TR 18037 7.18a.6, not re-implementations. Domains are 65536
// values; ESBMC proves rather than enumerates, so the width is not the cost
// driver here.
//
// divi/idiv/sqrt/exp for these types are handled in their own harnesses.
#include "src/__support/fixed_point/fx_bits.h"
#include "hdr/stdint_proxy.h"

extern "C" long _Fract nondet_lfract();
extern "C" unsigned long _Fract nondet_ulfract();
extern "C" _Accum nondet_accum();
extern "C" unsigned _Accum nondet_uaccum();
extern "C" int nondet_int();
extern "C" uint32_t nondet_u32();
extern "C" int32_t nondet_s32();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

namespace fx = LIBC_NAMESPACE::fixed_point;
using LIBC_NAMESPACE::cpp::bit_cast;

int main()
{
  /* ---------- abs: abslr (s0.15) and absk (s8.7) ---------- */
  {
    long _Fract x = nondet_lfract();
    long _Fract r = fx::abs(x);
    using R = fx::FXRep<long _Fract>;
    __ESBMC_assert(r >= R::ZERO(), "abslr: never negative");
    __ESBMC_assert(
      x == R::MIN() ? r == R::MAX() : (r == x || r == -x),
      "abslr: magnitude preserved, MIN saturates");
  }
  {
    _Accum x = nondet_accum();
    _Accum r = fx::abs(x);
    using R = fx::FXRep<_Accum>;
    __ESBMC_assert(r >= R::ZERO(), "absk: never negative");
    __ESBMC_assert(
      x == R::MIN() ? r == R::MAX() : (r == x || r == -x),
      "absk: magnitude preserved, MIN saturates");
  }

  /* ---------- countls: TR contract on all four types ---------- */
  {
    unsigned long _Fract v = nondet_ulfract();
    __ESBMC_assume(v != 0.0ulr);
    int k = fx::countls(v);
    __ESBMC_assert(k >= 0 && k <= 32, "countlsulr in range");
    __ESBMC_assert(
      ((unsigned long _Fract)(v << k) >> k) == v, "countlsulr: v<<k does not overflow");
    __ESBMC_assume(k < 32);
    __ESBMC_assert(
      ((unsigned long _Fract)(v << (k + 1)) >> (k + 1)) != v,
      "countlsulr: k is maximal");
  }
  {
    _Accum v = nondet_accum();
    __ESBMC_assume(v != 0.0k);
    int k = fx::countls(v);
    __ESBMC_assert(k >= 0 && k <= 31, "countlsk in range");
    __ESBMC_assert(
      ((_Accum)(v << k) >> k) == v, "countlsk: v<<k does not overflow");
  }
  {
    unsigned _Accum v = nondet_uaccum();
    __ESBMC_assume(v != 0.0uk);
    int k = fx::countls(v);
    __ESBMC_assert(k >= 0 && k <= 32, "countlsuk in range");
    __ESBMC_assert(
      ((unsigned _Accum)(v << k) >> k) == v,
      "countlsuk: v<<k does not overflow");
  }

  /* ---------- round: s0.15 and the first _Accum (s8.7) ---------- */
  {
    long _Fract x = nondet_lfract();
    int n = nondet_int();
    __ESBMC_assume(n >= 0 && n < 31);
    long _Fract r = fx::round(x, n);
    int32_t xr = bit_cast<int32_t, long _Fract>(x);
    int32_t rr = bit_cast<int32_t, long _Fract>(r);
    /* The bracket arithmetic must not overflow at ANY width: xr is a full
     * raw value and step reaches 2^(FRACTION_LEN), so `int` underflows at 32
     * bits (seen at xr = -1332312064, n = 1, where the harness wrongly
     * accused the library) and int64_t would underflow at 64. __int128 holds
     * every case with room to spare, so the class of bug cannot recur. */
    const __int128 step = (__int128)1 << (31 - n);
    if (rr != 2147483647)
    {
      __ESBMC_assert(rr % step == 0, "roundlr: multiple of 2^-n");
      const __int128 x128 = xr;
      const __int128 down =
        (x128 >= 0 ? (x128 / step) * step : ((x128 - step + 1) / step) * step);
      const __int128 up = down + step;
      if (up - x128 <= x128 - down)
        __ESBMC_assert(rr == up, "roundlr: ties round toward +Inf");
      else
        __ESBMC_assert(rr == down, "roundlr: nearer-down rounds down");
    }
  }
  {
    _Accum x = nondet_accum();
    int n = nondet_int();
    __ESBMC_assume(n >= 0 && n < 15); // FRACTION_LEN == 15 for s16.15
    _Accum r = fx::round(x, n);
    int32_t xr = bit_cast<int32_t, _Accum>(x);
    int32_t rr = bit_cast<int32_t, _Accum>(r);
    const __int128 step = (__int128)1 << (15 - n);
    if (rr != 2147483647)
    {
      __ESBMC_assert(rr % step == 0, "roundk: multiple of 2^-n");
      const __int128 x128 = xr;
      const __int128 down =
        (x128 >= 0 ? (x128 / step) * step : ((x128 - step + 1) / step) * step);
      const __int128 up = down + step;
      if (up - x128 <= x128 - down)
        __ESBMC_assert(rr == up, "roundk: ties round toward +Inf");
      else
        __ESBMC_assert(rr == down, "roundk: nearer-down rounds down");
    }
  }

  /* ---------- bits/fxbits round-trips on all four types ---------- */
  {
    long _Fract f = nondet_lfract();
    __ESBMC_assert(
      bit_cast<long _Fract, int32_t>(bit_cast<int32_t, long _Fract>(f)) == f,
      "lrbits(bitslr(f)) == f");
    unsigned long _Fract u = nondet_ulfract();
    __ESBMC_assert(
      bit_cast<unsigned long _Fract, uint32_t>(bit_cast<uint32_t, unsigned long _Fract>(
        u)) == u,
      "ulrbits(bitsulr(u)) == u");
    _Accum a = nondet_accum();
    __ESBMC_assert(
      bit_cast<_Accum, int32_t>(bit_cast<int32_t, _Accum>(a)) == a,
      "kbits(bitsk(a)) == a");
    unsigned _Accum ua = nondet_uaccum();
    __ESBMC_assert(
      bit_cast<unsigned _Accum, uint32_t>(
        bit_cast<uint32_t, unsigned _Accum>(ua)) == ua,
      "ukbits(bitsuk(ua)) == ua");
  }

  return 0;
}
