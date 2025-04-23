#include <assert.h>
#include <stdint.h>
#include <stdbool.h>

DEFINE_ESBMC_OVERFLOW_TYPE(int)

int main() {
  int a = __VERIFIER_nondet_int();
  int b = __VERIFIER_nondet_int();

  __ESBMC_overflow_result foo;

  // Test addition
  foo = __ESBMC_overflow_result_plus(a, b);
  assert(!foo.overflow);

  // Test subtraction
  foo = __ESBMC_overflow_result_minus(a, b);
  assert(!foo.overflow);

  // Test multiplication
  foo = __ESBMC_overflow_result_mult(a, b);
  assert(!foo.overflow);

  // Test shift left
  foo = __ESBMC_overflow_result_shl(a, b);
  assert(!foo.overflow);

  // Test unary minus
  foo = __ESBMC_overflow_result_unary_minus(a);
  assert(!foo.overflow);

  return 0;
}

