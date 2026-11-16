#include <limits.h>
unsigned nondet_uint(void);
int main(void)
{
  unsigned a = nondet_uint(), b = nondet_uint();
  if (a <= UINT_MAX - b)
    return a + b;
  return 0;
}
