/* github #807: integer-to-float conversions under --fp2bv agree with an
 * integer-arithmetic reference in every rounding mode, and a rounding carry
 * out of the largest exponent gives infinity. */
#include <assert.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

int nondet_int(void);
unsigned __int128 nondet_u128(void);
unsigned _BitInt(25) nondet_b25(void);

static int msb(uint32_t t)
{
  int e = 0;
  if (t >> 16) { e += 16; t >>= 16; }
  if (t >> 8) { e += 8; t >>= 8; }
  if (t >> 4) { e += 4; t >>= 4; }
  if (t >> 2) { e += 2; t >>= 2; }
  if (t >> 1) e += 1;
  return e;
}

static uint32_t ref(uint32_t m, int neg, int mode)
{
  int e = msb(m);
  uint32_t bits;
  if (e <= 23)
    bits = ((uint32_t)(e + 127) << 23) | ((m << (23 - e)) & 0x7fffff);
  else
  {
    int sh = e - 23;
    uint32_t sig = m >> sh;
    uint32_t rest = m & ((1u << sh) - 1);
    uint32_t half = 1u << (sh - 1);
    int inc = mode == FE_TONEAREST ? rest > half || (rest == half && (sig & 1))
              : mode == FE_UPWARD   ? !neg && rest
              : mode == FE_DOWNWARD ? neg && rest
                                    : 0;
    bits = (((uint32_t)(e + 127) << 23) | (sig & 0x7fffff)) + inc;
  }
  return bits | ((uint32_t)neg << 31);
}

int main(void)
{
  int modes[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
  int mode = modes[nondet_int() & 3];
  fesetround(mode);

  int x = nondet_int();
  __ESBMC_assume(x != 0);
  float f = (float)x;
  uint32_t b;
  memcpy(&b, &f, sizeof b);
  int neg = x < 0;
  assert(b == ref(neg ? -(uint32_t)x : (uint32_t)x, neg, mode));

  double dc = (signed char)x, dx = x;
  assert(dc > -129.0 && dc < 128.0 && (x > 0 ? dx >= 1.0 : dx <= -1.0));

  fesetround(FE_TONEAREST);
  unsigned _BitInt(25) v = nondet_b25();
  __ESBMC_assume(v == 0x1000003 || v == 0x1000001);
  assert((float)v == (v == 0x1000003 ? 16777220.0f : 16777216.0f));

  unsigned __int128 big = nondet_u128();
  __ESBMC_assume(big >= ~(unsigned __int128)0 - 1000);
  assert(isinf((float)big));
  fesetround(FE_TOWARDZERO);
  assert((float)big == FLT_MAX);
  return 0;
}
