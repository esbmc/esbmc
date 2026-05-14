#include <assert.h>

int nondet_int(void);

int main()
{
  int x = nondet_int();
  int y = nondet_int();
  int c1 = 3, c2 = 4;

  // LLVM-style reassoc: A + (B + C) -> (A + B) + C when A+B simplifies.
  // c1 + (x + c2): commute_binop moves c1 to the right, giving (x + c2) + c1
  // = (A=x, B=c2) + C=c1.  Try B+C = c2+c1 = 7 (constant fold).
  // Result: A + V = x + 7.
  assert((c1 + (x + c2)) == (x + 7));

  // (A + B) + C with C = -A simplifies via the commutative form:
  // (A=x, B=y) + C=(-x).  Try C+A = (-x)+x = 0.
  // Result: V + B = 0 + y = y.
  assert(((x + y) + (-x)) == y);

  // Symmetric: A + (B + C) where C cancels A.
  // Frontend rewrites (-x) + (...) -> (...) - x for the outer add, so
  // probe the same shape from the right side.
  assert((y + ((-y) + x)) == x);

  return 0;
}
