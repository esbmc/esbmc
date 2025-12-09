#include <assert.h>

int main() {
  int a = nondet_int();
  int b = nondet_int();

  __ESBMC_assume(a > 0 && b > 0);

  assert((a * b) > 0);
}
