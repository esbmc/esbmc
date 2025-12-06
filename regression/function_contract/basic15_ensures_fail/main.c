/* Basic15_ensures_fail: Large number violation
 * This should FAIL - violates ensures with large numbers
 */
#include <assert.h>

int add_large(int a, int b)
{


  __ESBMC_ensures(__ESBMC_return_value == a + b);

  // VIOLATION: returns wrong sum
  return a + b + 1;  // Should be a + b
}

int main()
{



  int result = add_large(1000, 2000);
  return 0;
}

