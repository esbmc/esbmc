/* Test __ESBMC_old with double return type - should FAIL
 * Similar to basic24 but tests __ESBMC_old with floating point
 */

double global_value = 0.0;

double multiply_and_update(double x, double y)
{
  __ESBMC_requires(x > 0.0 && y > 0.0);
  __ESBMC_ensures(__ESBMC_return_value == x * y);
  __ESBMC_ensures(global_value == __ESBMC_old(global_value) + __ESBMC_return_value);
  
  double result = x * y;
  // BUG: Should add result, but adds result + 1.0
  global_value += (result + 1.0);
  return result;
}

int main()
{
  global_value = 10.0;
  double result = multiply_and_update(2.5, 4.0);
  // Result should be 10.0, global should be 20.0
  // But global will actually be 21.0 due to bug
  return 0;
}

