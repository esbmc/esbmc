/* Basic7_ensures_fail: Ensures violation with multiple parameters
 * This should FAIL verification
 */
#include <assert.h>

int max(int a, int b)
{


  __ESBMC_ensures(__ESBMC_return_value >= a);
  __ESBMC_ensures(__ESBMC_return_value >= b);

  // VIOLATION: returns smaller value instead of larger
  return (a < b) ? a : b;
}

int main()
{



  int x = 10;
  int y = 20;
  int result = max(x, y);
  
  return 0;
}

