#include <assert.h>

int main() 
{
  double x = 1e-200, z = 1e+200;
  double result3 = x / z;        // Underflows to 0.0
  assert(result3 > 0.0);         // FAILS - underflow
  return 0;
}
