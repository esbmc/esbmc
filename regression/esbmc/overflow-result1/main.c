#include <assert.h>
#include <stdint.h>
#include <stdbool.h>

DEFINE_ESBMC_OVERFLOW_TYPE(int)

int main() {
  int a = __VERIFIER_nondet_int();
  int b = __VERIFIER_nondet_int();

  __ESBMC_assume(a >= -1000 && a <= 1000);
  __ESBMC_assume(b >= -1000 && b <= 1000);

  __ESBMC_overflow_result foo;

  foo = __ESBMC_overflow_result_plus(a, b);
  
  if (!foo.overflow)
    assert(foo.result == a + b);

  return 0;
}


