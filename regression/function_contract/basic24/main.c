/* Basic24: Test __ESBMC_return_value with different return types
 * Tests that __ESBMC_return_value type matches function return type
 * (int, double, etc.)
 */
#include <assert.h>

int increment(int x)
{
  __ESBMC_requires(x > 0);
  // Test __ESBMC_return_value with int return type
  __ESBMC_ensures(__ESBMC_return_value > x);
  __ESBMC_ensures(__ESBMC_return_value == x + 1);
  return x + 1;
}

double multiply_double(double x, double y)
{
  __ESBMC_requires(x > 0 && y > 0);
  // Test __ESBMC_return_value with double return type
  __ESBMC_ensures(__ESBMC_return_value > 0);
  __ESBMC_ensures(__ESBMC_return_value == x * y);
  return x * y;
}

float divide_float(float x, float y)
{
  __ESBMC_requires(y != 0.0f);
  // Test __ESBMC_return_value with float return type
  __ESBMC_ensures(__ESBMC_return_value == x / y);
  return x / y;
}
 
int main()
{
  // Test int return type with __ESBMC_return_value
  int a = 5;
  int result = increment(a);
  assert(result > a);
  assert(result == 6);
  
  // Test double return type with __ESBMC_return_value
  double d1 = 2.5, d2 = 3.0;
  double d_result = multiply_double(d1, d2);
  assert(d_result == 7.5);
  
  // Test float return type with __ESBMC_return_value
  float f1 = 10.0f, f2 = 2.0f;
  float f_result = divide_float(f1, f2);
  assert(f_result == 5.0f);
  
  return 0;
}

