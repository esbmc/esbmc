// 16-bit tier of LLVM libc's stdfix: the four types with 16-bit storage,
//
//   _Fract (s0.15)   unsigned _Fract (u0.16)
//   short _Accum (s8.7)   unsigned short _Accum (u8.8)
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

extern "C" _Fract nondet_fract();
extern "C" unsigned _Fract nondet_ufract();
extern "C" short _Accum nondet_saccum();
extern "C" unsigned short _Accum nondet_usaccum();
extern "C" int nondet_int();
extern "C" uint16_t nondet_u16();
extern "C" int16_t nondet_s16();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

namespace fx = LIBC_NAMESPACE::fixed_point;
using LIBC_NAMESPACE::cpp::bit_cast;

int main()
{
  /* ---------- abs: absr (s0.15) and abshk (s8.7) ---------- */
  {
    _Fract x = nondet_fract();
    _Fract r = fx::abs(x);
    using R = fx::FXRep<_Fract>;
    __ESBMC_assert(r >= R::ZERO(), "absr: never negative");
    __ESBMC_assert(
      x == R::MIN() ? r == R::MAX() : (r == x || r == -x),
      "absr: magnitude preserved, MIN saturates");
  }
  {
    short _Accum x = nondet_saccum();
    short _Accum r = fx::abs(x);
    using R = fx::FXRep<short _Accum>;
    __ESBMC_assert(r >= R::ZERO(), "abshk: never negative");
    __ESBMC_assert(
      x == R::MIN() ? r == R::MAX() : (r == x || r == -x),
      "abshk: magnitude preserved, MIN saturates");
  }

  /* ---------- countls: TR contract on all four types ---------- */
  {
    unsigned _Fract v = nondet_ufract();
    __ESBMC_assume(v != 0.0ur);
    int k = fx::countls(v);
    __ESBMC_assert(k >= 0 && k <= 16, "countlsur in range");
    __ESBMC_assert(
      ((unsigned _Fract)(v << k) >> k) == v, "countlsur: v<<k does not overflow");
    __ESBMC_assume(k < 16);
    __ESBMC_assert(
      ((unsigned _Fract)(v << (k + 1)) >> (k + 1)) != v,
      "countlsur: k is maximal");
  }
  {
    short _Accum v = nondet_saccum();
    __ESBMC_assume(v != 0.0hk);
    int k = fx::countls(v);
    __ESBMC_assert(k >= 0 && k <= 15, "countlshk in range");
    __ESBMC_assert(
      ((short _Accum)(v << k) >> k) == v, "countlshk: v<<k does not overflow");
  }
  {
    unsigned short _Accum v = nondet_usaccum();
    __ESBMC_assume(v != 0.0uhk);
    int k = fx::countls(v);
    __ESBMC_assert(k >= 0 && k <= 16, "countlsuhk in range");
    __ESBMC_assert(
      ((unsigned short _Accum)(v << k) >> k) == v,
      "countlsuhk: v<<k does not overflow");
  }

  /* ---------- round: s0.15 and the first _Accum (s8.7) ---------- */
  {
    _Fract x = nondet_fract();
    int n = nondet_int();
    __ESBMC_assume(n >= 0 && n < 15);
    _Fract r = fx::round(x, n);
    int16_t xr = bit_cast<int16_t, _Fract>(x);
    int16_t rr = bit_cast<int16_t, _Fract>(r);
    const __int128 step = (__int128)1 << (15 - n);
    if (rr != 32767)
    {
      __ESBMC_assert(rr % step == 0, "roundr: multiple of 2^-n");
      const __int128 x128 = xr;
      const __int128 down =
        (x128 >= 0 ? (x128 / step) * step : ((x128 - step + 1) / step) * step);
      const __int128 up = down + step;
      if (up - x128 <= x128 - down)
        __ESBMC_assert(rr == up, "roundr: ties round toward +Inf");
      else
        __ESBMC_assert(rr == down, "roundr: nearer-down rounds down");
    }
  }
  {
    short _Accum x = nondet_saccum();
    int n = nondet_int();
    __ESBMC_assume(n >= 0 && n < 7); // FRACTION_LEN == 7 for s8.7
    short _Accum r = fx::round(x, n);
    int16_t xr = bit_cast<int16_t, short _Accum>(x);
    int16_t rr = bit_cast<int16_t, short _Accum>(r);
    const __int128 step = (__int128)1 << (7 - n);
    if (rr != 32767)
    {
      __ESBMC_assert(rr % step == 0, "roundhk: multiple of 2^-n");
      const __int128 x128 = xr;
      const __int128 down =
        (x128 >= 0 ? (x128 / step) * step : ((x128 - step + 1) / step) * step);
      const __int128 up = down + step;
      if (up - x128 <= x128 - down)
        __ESBMC_assert(rr == up, "roundhk: ties round toward +Inf");
      else
        __ESBMC_assert(rr == down, "roundhk: nearer-down rounds down");
    }
  }

  /* ---------- bits/fxbits round-trips on all four types ---------- */
  {
    _Fract f = nondet_fract();
    __ESBMC_assert(
      bit_cast<_Fract, int16_t>(bit_cast<int16_t, _Fract>(f)) == f,
      "rbits(bitsr(f)) == f");
    unsigned _Fract u = nondet_ufract();
    __ESBMC_assert(
      bit_cast<unsigned _Fract, uint16_t>(bit_cast<uint16_t, unsigned _Fract>(
        u)) == u,
      "urbits(bitsur(u)) == u");
    short _Accum a = nondet_saccum();
    __ESBMC_assert(
      bit_cast<short _Accum, int16_t>(bit_cast<int16_t, short _Accum>(a)) == a,
      "hkbits(bitshk(a)) == a");
    unsigned short _Accum ua = nondet_usaccum();
    __ESBMC_assert(
      bit_cast<unsigned short _Accum, uint16_t>(
        bit_cast<uint16_t, unsigned short _Accum>(ua)) == ua,
      "uhkbits(bitsuhk(ua)) == ua");
  }

  return 0;
}
