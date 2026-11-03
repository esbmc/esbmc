#include <assert.h>
int nondet_int(void);

int inc(int x) { return x + 1; }

int main(void) {
  int k = nondet_int();
  __ESBMC_assume(k > 0 && k < 10);
  int l = inc(5);
  assert(k < 10);
  return l - 6;
}
