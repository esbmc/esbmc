/*
 * According with the section "Bitwise shift operators" of the C99 ISO specification:
 * "If the value of the right operand is negative or is greater than or equal to the
    width of the promoted left operand, the behavior is undefined."
 */

#include <limits.h>

int main() {
  const int shift = CHAR_BIT * sizeof(int);
  const int a = 1;
  int b = a << shift; // right operand is equal to the width of the left operand
  return 0;
}
