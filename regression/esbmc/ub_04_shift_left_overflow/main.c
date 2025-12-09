/*
 * The section "Bitwise shift operators" of the C99 ISO specifies the following about shift left:
 * If E1 has a signed type and nonnegative value, and E1 × 2^E2 is representable in the result type,
 * then that is the resulting value; otherwise, the behavior is undefined.
 */

#include <limits.h>

int main()
{
  int e1 = INT_MAX;
  int e2 = 10;
  int v = e1 << e2; // E1 × 2^E2 is not representable in the result type
  return 0;
}
