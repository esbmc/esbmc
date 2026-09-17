#include <stdint.h>
int main()
{
  float big = 1.0e30f;
  double neg = -1.0e30;
  int8_t a = (int8_t)big;    /* out of range: UB in C */
  int64_t b = (int64_t)neg;  /* out of range */
  __CPROVER_assert(a == a, "no crash converting out of range");
  __CPROVER_assert(b == b, "no crash converting negative out of range");
  return 0;
}
