#include <assert.h>

double fma(double, double);

int main(void)
{
  assert(fma(2.0, 3.0) == 5.0);
  return 0;
}
