/*
 * The simplifier folds `if(c, x, x)` to `x` (algebraic), erasing `c`
 * from the value expression. ESBMC's --overflow-check inserts the
 * overflow assertion at goto-program time as a separate VCC, BEFORE
 * any simplifier pass touches the value expression. So even when the
 * if-fold drops the condition, the overflow check survives.
 *
 * Pin contract: an if-condition with signed overflow, with both arms
 * identical, must still report VERIFICATION FAILED under
 * --overflow-check.
 */
#include <limits.h>
int nondet_int();

int main() {
  int x = nondet_int();
  int y = nondet_int();
  __ESBMC_assume(x == INT_MAX);
  __ESBMC_assume(y == 1);
  // x + y overflows; both arms identical so simplifier folds the if.
  int v = (x + y > 0) ? 42 : 42;
  return v;
}
