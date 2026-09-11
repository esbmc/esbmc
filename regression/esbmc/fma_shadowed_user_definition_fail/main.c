#include <assert.h>

double fma(double a, double b, double c)
{
  return 42.0;
}

int main(void)
{
  assert(fma(2.0, 3.0, 4.0) == 10.0);
  return 0;
}
