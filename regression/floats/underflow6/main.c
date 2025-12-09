#include <assert.h>

int main()
{
  double a = 1e-200, b = 1e-200;
  double result3 = a * b;       // Underflows to 0.0
  assert(result3 > 0.0);        // FAILS - double precision underflow
  return 0;
}

