#include <assert.h>

int main() 
{
  double x = 1e-200, y = 1e-200;
  double result1 = x * y;        // Underflows to 0.0
  assert(result1 > 0.0);         // FAILS - underflow
  return 0;
}
