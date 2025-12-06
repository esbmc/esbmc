/* Example 5: Using --function with --enforce-contract
 * Demonstrates how to combine --function and --enforce-contract options
 */
#include <assert.h>

int increment(int x)
{


  __ESBMC_requires(x > 0);
  __ESBMC_ensures(__ESBMC_return_value > x);

  return x + 1;
}

int decrement(int x)
{


  __ESBMC_requires(x > 0);
  __ESBMC_ensures(__ESBMC_return_value < x);

  return x - 1;
}

// Custom main function (not named "main")
int my_main()
{



  int a = 10;
  int result1 = increment(a);
  assert(result1 > a);
  
  int result2 = decrement(a);
  assert(result2 < a);
  
  return 0;
}

// Standard main function
int main()
{



  return my_main();
}

