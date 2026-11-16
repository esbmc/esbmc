#include <assert.h>

extern int __VERIFIER_nondet_int(void);

struct pair
{
  int a, b;
};

struct pair unused, used;

union word
{
  int i;
  float f;
} w;

int main()
{
  /* The SUCCESSFUL twin of testcomp_gen_aggregate_literal: same writes, but
   * the assertion is unreachable, so no test case is written. */
  w.i = __VERIFIER_nondet_int();
  unused.a = __VERIFIER_nondet_int();
  unused.b = __VERIFIER_nondet_int();
  used.b = __VERIFIER_nondet_int();
  used.a = __VERIFIER_nondet_int();
  if (used.a == 1 && used.b == 2 && used.a == used.b)
    assert(0);
  return 0;
}
