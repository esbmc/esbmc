/*
 * Pin contract: the simplifier rewrites `x >= 0 ? x : -x` to `abs(x)` for
 * shape canonicalization, but ESBMC's --overflow-check inserts the neg
 * overflow assertion at goto-program time, BEFORE symex/simplifier run.
 * That assertion survives the rewrite, so x = INT_MIN still fails under
 * --overflow-check.
 */
#include <limits.h>
int nondet_int();

int main() {
  int x = nondet_int();
  __ESBMC_assume(x == INT_MIN);
  // -INT_MIN overflows. The cond's negation arm is exercised when x < 0.
  int y = (x >= 0) ? x : -x;
  return y;
}
