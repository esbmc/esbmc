/* #10642 with an ordinary guarded assertion: the invariant state x == 1
 * violates it, x == 0 skips it, so the violation is the abstraction's. */
#include <assert.h>
extern int __VERIFIER_nondet_int(void);
int main() {
  int x = 0, y = 0;
  __ESBMC_loop_invariant(0 <= x && x <= 1 && 0 <= y && y <= 1);
  while (__VERIFIER_nondet_int()) {
    if (x == 1)
      assert(y == 5);
    x = 0;
  }
  return 0;
}
