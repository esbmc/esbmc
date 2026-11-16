#include <assert.h>
int nondet_int(void);

int helper(int x) { return x; }

void other(int n) {
  if (n > 0)
    other(n - 1);
}

/* rec, other and helper sit in three different call-graph components, so
 * returning from other or helper must keep rec's own l */
void rec(int n) {
  if (n > 0)
    rec(n - 1);
  int l = nondet_int();
  __ESBMC_assume(l > 0 && l < 10);
  other(1);
  helper(1);
  assert(l < 10);
}

int main(void) {
  rec(2);
  return 0;
}
