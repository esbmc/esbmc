// __builtin_ffs/ffsl/ffsll had no model, so symex gave the call a nondet
// return and any computation over it silently got garbage. Check exact values
// against a loop reference, not just a range: a nondet result constrained to
// the right interval would still be wrong. #183
#include <assert.h>

unsigned nondet_uint(void);

static int ref_ffs(unsigned v)
{
  for (int i = 0; i < 32; i++)
    if (v & (1u << i))
      return i + 1;
  return 0;
}

int main(void)
{
  unsigned v = nondet_uint();
  assert(__builtin_ffs((int)v) == ref_ffs(v));

  // Zero is defined for ffs, unlike clz and ctz.
  assert(__builtin_ffs(0) == 0);
  assert(__builtin_ffsl(0L) == 0);
  assert(__builtin_ffsll(0LL) == 0);

  assert(__builtin_ffs(1) == 1);
  assert(__builtin_ffs(8) == 4);
  assert(__builtin_ffs(-1) == 1);
  assert(__builtin_ffs((int)0x80000000u) == 32);

  // The index is one-based and taken at the operand's own width.
  assert(__builtin_ffsl(1L << 20) == 21);
#if __SIZEOF_LONG__ >= 8
  // Only where long is 64-bit: LLP64 (Windows) would shift past its width.
  assert(__builtin_ffsl(1L << 40) == 41);
#endif
  assert(__builtin_ffsll(1LL << 60) == 61);

  // ffs is ctz + 1 wherever ctz is defined.
  unsigned u = nondet_uint();
  __ESBMC_assume(u != 0);
  assert(__builtin_ffs((int)u) == __builtin_ctz(u) + 1);

  return 0;
}
