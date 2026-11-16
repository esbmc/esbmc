#include <assert.h>

extern int __VERIFIER_nondet_int(void);

struct pair
{
  int a, b;
} g;

int x;

int main()
{
  /* With c == 0, the call writing g.a never runs: its literal
   * `g == { .a=nondet2, .b=0 }` sits under a false guard and must not become
   * an input. */
  int c = __VERIFIER_nondet_int();
  if (c)
  {
    x = __VERIFIER_nondet_int();
    g.a = __VERIFIER_nondet_int();
  }
  else
  {
    x = __VERIFIER_nondet_int();
    g.b = __VERIFIER_nondet_int();
  }
  int d = __VERIFIER_nondet_int();
  if (!c && x == 3 && g.b == 4 && d == 5)
    assert(0);
  return 0;
}
