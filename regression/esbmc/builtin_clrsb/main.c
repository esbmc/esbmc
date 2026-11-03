// __builtin_clrsb/clrsbl/clrsbll had no model, so the call returned nondet and
// clrsb(-1) == 31 failed. Check exact values against a loop reference.
// --clz-zero-check must leave clrsb alone: it is defined at zero.
#include <assert.h>

int nondet_int(void);

static int ref_clrsb(int x)
{
  unsigned u = (unsigned)x, sign = u >> 31;
  int n = 0;
  while (n < 31 && ((u >> (30 - n)) & 1u) == sign)
    n++;
  return n;
}

int main(void)
{
  int x = nondet_int();
  assert(__builtin_clrsb(x) == ref_clrsb(x));

  assert(__builtin_clrsb(0) == 31);
  assert(__builtin_clrsb(-1) == 31);
  assert(__builtin_clrsb(1) == 30);
  assert(__builtin_clrsb(-2) == 30);
  assert(__builtin_clrsb((int)0x80000000u) == 0);
  assert(__builtin_clrsb(0x40000000) == 0);

  // The count is taken at the operand's own width.
  assert(__builtin_clrsbl(0L) == (int)(8 * sizeof(long)) - 1);
  assert(__builtin_clrsbll(-1LL) == 63);
  assert(__builtin_clrsbll(1LL << 40) == 22);
  return 0;
}
