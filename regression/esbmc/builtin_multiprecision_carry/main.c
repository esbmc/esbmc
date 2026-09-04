#include <assert.h>

unsigned nondet_uint(void);

/* __builtin_addc/__builtin_subc return the modular result and store whether
 * either partial step wrapped. Both steps need checking: a + b may fit and
 * adding the carry then wrap. Unmodelled, these returned nondet, which is how
 * llvm-libc's add_with_carry silently produced a zero sum. */
int main(void)
{
  unsigned a = nondet_uint(), b = nondet_uint(), cin = nondet_uint();
  __ESBMC_assume(cin <= 1);

  unsigned cout;
  unsigned s = __builtin_addc(a, b, cin, &cout);
  unsigned long long exact = (unsigned long long)a + b + cin;
  assert(s == (unsigned)exact);
  assert(cout == (unsigned)(exact >> 32));

  /* The carry-out of the second step alone, which a single check would miss. */
  unsigned c2;
  assert(__builtin_addc(0xffffffffu, 0u, 1u, &c2) == 0u && c2 == 1u);

  unsigned long c3;
  assert(__builtin_subcl(0UL, 1UL, 0UL, &c3) == ~0UL && c3 == 1UL);

  return 0;
}
