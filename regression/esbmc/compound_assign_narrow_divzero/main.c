#include <assert.h>

/* The other direction of the same defect: narrowing the right operand of `/=`
   to char turns the divisor 256 into 0, so a division that is well defined in C
   (100 / 256 == 0) is reported as a division by zero. */
int main()
{
  char b = 100;
  int a = 256;
  b /= a;
  assert(b == 0);
  return 0;
}
