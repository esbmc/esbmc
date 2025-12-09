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

  // Test addition
  foo = __ESBMC_overflow_result_plus(a, b);
  if (!foo.overflow)
    assert(foo.result == a + b);

  // Test subtraction
  foo = __ESBMC_overflow_result_minus(a, b);
  if (!foo.overflow)
    assert(foo.result == a - b);

  // Test multiplication
  foo = __ESBMC_overflow_result_mult(a, b);
  if (!foo.overflow)
    assert(foo.result == a * b);

  // Test shift left
  foo = __ESBMC_overflow_result_shl(a, b);
  if (!foo.overflow)
    assert(foo.result == (a << b));

  // Test unary minus
  foo = __ESBMC_overflow_result_unary_minus(a);
  if (!foo.overflow)
    assert(foo.result == -a);

  return 0;
}

