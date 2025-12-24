/* Basic16_ensures_fail: Comparison operator violation
 * This should FAIL - violates ensures with comparison operators
 */
#include <assert.h>

int compare(int a, int b)
{


  __ESBMC_ensures(__ESBMC_return_value >= -1);
  __ESBMC_ensures(__ESBMC_return_value <= 1);

  // VIOLATION: returns value outside [-1, 1] range
  return 2;  // Should be -1, 0, or 1
}

int main()
{



  int result = compare(5, 10);
  return 0;
}

