#include <assert.h>

int nondet_int(void);

// Sub-heavy chain on two int args. GOTO reassoc canonicalizes the constants
// among themselves, but until symex substitutes the args it can't fold across
// them. SSA reassoc fires after substitution and produces the canonical form.
static int compute(int a, int b)
{
  return 5 - a - 8 - b - 2; // body shape: -5 - a - b
}

int main()
{
  int x = nondet_int();
  int y = nondet_int();

  // First arg concrete, second nondet: after symex inlines compute(10, x)
  // the body is "5 - 10 - 8 - x - 2", which SSA reassoc folds to "-15 - x".
  int r1 = compute(10, x);
  assert(r1 == -15 - x);

  // Both args nondet: the chain stays symbolic but the constants among
  // themselves should still combine: "-5 - x - y".
  int r2 = compute(x, y);
  assert(r2 == -5 - x - y);

  // Both concrete: the body is fully constant; assert holds trivially.
  int r3 = compute(100, 200);
  assert(r3 == -305);

  return 0;
}
