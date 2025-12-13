/* Test: Multiple ensures clauses - one violated
 * This should FAIL verification because one ensures clause is violated
 */
#include <assert.h>

int clamp(int x, int min_val, int max_val)
{


  __ESBMC_requires(min_val <= max_val);
  __ESBMC_ensures(__ESBMC_return_value >= min_val);
  __ESBMC_ensures(__ESBMC_return_value <= max_val);

  // VIOLATION: This implementation violates the ensures clauses
  if (x < min_val)
    return min_val - 1;  // Should return min_val
  if (x > max_val)
    return max_val + 1;  // Should return max_val
  return x;
}

int main()
{



  int result1 = clamp(15, 0, 10);  // Should return 10, but returns 11
  int result2 = clamp(-5, 0, 10);  // Should return 0, but returns -1
  
  return 0;
}

