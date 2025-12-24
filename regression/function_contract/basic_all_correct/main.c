/* Test: Correct implementation - should PASS
 * All contract clauses are satisfied
 */
#include <assert.h>

int increment(int x)
{
  __ESBMC_requires(x > 0);
  __ESBMC_ensures(__ESBMC_return_value > x);

  return x + 1;  // CORRECT: satisfies ensures clause
}

int divide(int x, int y)
{


  __ESBMC_requires(x >= 0);
  __ESBMC_requires(y > 0);
  __ESBMC_ensures(__ESBMC_return_value >= 0);
  __ESBMC_ensures(__ESBMC_return_value <= x);

  return x / y;  // CORRECT: satisfies all ensures clauses
}

int clamp(int x, int min_val, int max_val)
{


  __ESBMC_requires(min_val <= max_val);
  __ESBMC_ensures(__ESBMC_return_value >= min_val);
  __ESBMC_ensures(__ESBMC_return_value <= max_val);

  if (x < min_val)
    return min_val;  // CORRECT
  if (x > max_val)
    return max_val;  // CORRECT
  return x;  // CORRECT
}

int main()
{



  int a = 5;
  int result1 = increment(a);
  assert(result1 > a);
  
  int result2 = divide(10, 3);
  assert(result2 >= 0);
  assert(result2 <= 10);
  
  int result3 = clamp(15, 0, 10);
  assert(result3 >= 0);
  assert(result3 <= 10);
  
  return 0;
}

