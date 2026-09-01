int main() {
  int x = -2147483647 - 1;
  __CPROVER_assert(__CPROVER_overflow_unary_minus(x), "-INT_MIN overflows");
  int y = 5;
  __CPROVER_assert(!__CPROVER_overflow_unary_minus(y), "-5 does not overflow");
  return 0;
}
