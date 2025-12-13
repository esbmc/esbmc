/* Example 3: All comparison operators
 * Tests >=, <=, >, <, ==, != operators
 */
#include <assert.h>

int clamp(int x, int min_val, int max_val)
{


  __ESBMC_requires(min_val <= max_val);
  __ESBMC_ensures(__ESBMC_return_value >= min_val);
  __ESBMC_ensures(__ESBMC_return_value <= max_val);

  if (x < min_val)
    return min_val;
  if (x > max_val)
    return max_val;
  return x;
}

int max(int a, int b)
{


  __ESBMC_ensures(__ESBMC_return_value >= a);
  __ESBMC_ensures(__ESBMC_return_value >= b);
  __ESBMC_ensures(__ESBMC_return_value == a || __ESBMC_return_value == b);

  return (a > b) ? a : b;
}

int main()
{



  int result1 = clamp(15, 0, 10);
  assert(result1 >= 0);
  assert(result1 <= 10);
  
  int result2 = max(5, 8);
  assert(result2 >= 5);
  assert(result2 >= 8);
  
  return 0;
}

