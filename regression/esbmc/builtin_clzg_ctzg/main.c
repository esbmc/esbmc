// __builtin_clzg/ctzg and the ctz and 16-bit spellings were unmodelled, so
// symex gave the call a nondet return and any computation over it silently got
// garbage. Check exact values against a loop reference, not just a range: a
// nondet result constrained to the right interval would still be wrong. #6925.
#include <assert.h>
#include <stdint.h>

uint8_t nd8(void);
unsigned ndu(void);

static int ref_clz8(uint8_t v)
{
  int n = 0;
  for (int i = 7; i >= 0; i--)
  {
    if (v & (1u << i))
      break;
    n++;
  }
  return n;
}

static int ref_ctz8(uint8_t v)
{
  int n = 0;
  for (int i = 0; i < 8; i++)
  {
    if (v & (1u << i))
      break;
    n++;
  }
  return n;
}

int main(void)
{
  uint8_t v = nd8();
  assert(__builtin_clzg(v, 8) == ref_clz8(v));
  assert(__builtin_ctzg(v, 8) == ref_ctz8(v));

  // The width comes from the operand's own type, so a narrow argument is not
  // counted as if it had been promoted.
  assert(__builtin_clzg((uint8_t)0x10, 99) == 3);
  assert(__builtin_ctzg((uint8_t)0x10, 99) == 4);

  // The second argument is the result at zero, and is not the width in
  // disguise: it is returned verbatim, negative values included.
  assert(__builtin_clzg((uint8_t)0, 99) == 99);
  assert(__builtin_ctzg((uint8_t)0, -7) == -7);

  assert(__builtin_ctz(1u) == 0);
  assert(__builtin_ctz(0x80000000u) == 31);
  assert(__builtin_ctzll(1ull << 40) == 40);
  assert(__builtin_clzs((unsigned short)1) == 15);
  assert(__builtin_ctzs((unsigned short)0x8000) == 15);

  unsigned u = ndu();
  __ESBMC_assume(u != 0);
  assert(__builtin_clz(u) + __builtin_ctz(u) <= 31);
  return 0;
}
