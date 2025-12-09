/*
 * The section "Bitwise shift operators" of the C99 ISO specifies the following about shift left:
 * "If E1 has a signed type and nonnegative value, and E1 × 2^E2 is representable in the result type,
 * then that is the resulting value; otherwise, the behavior is undefined."
 */

int main() {
  int e1 = -2;
  int e2 = 1;
  int n = e1 << e2;  // E1 has a signed type and negative value
  return 0;
}
