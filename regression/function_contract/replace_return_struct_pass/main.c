/* Test: Struct return type with __ESBMC_return_value in replace mode - PASS case
 * Expected: VERIFICATION SUCCESSFUL (after __ESBMC_return_value type is fixed)
 * 
 * Current Status: Uses type casting syntax to work around Clang parsing limitation
 * This test uses ((Point*)&__ESBMC_return_value)->x syntax which Clang can parse.
 * In the conversion phase, we will recognize this pattern and replace it correctly.
 * 
 * Expected behavior (when fully implemented):
 * - Type casting syntax ((Point*)&__ESBMC_return_value)->x should be recognized
 * - __ESBMC_return_value should be replaced with actual return value variable
 * - Verification should succeed when contract is satisfied
 */
#include <assert.h>

typedef struct {
  int x;
  int y;
} Point;

Point make_point(int x, int y)
{
  __ESBMC_requires(x >= 0);
  __ESBMC_requires(y >= 0);
  // Use type casting syntax: ((Point*)&__ESBMC_return_value)->member
  // This allows Clang to parse successfully, and we'll handle it in conversion phase
  __ESBMC_ensures(((Point*)&__ESBMC_return_value)->x == x);
  __ESBMC_ensures(((Point*)&__ESBMC_return_value)->y == y);
  
  Point p;
  p.x = x;
  p.y = y;
  return p;
}

int main()
{
  int a = 5;
  int b = 10;
  Point result = make_point(a, b);  // Call replaced with contract
  
  // After contract replacement (when fixed):
  // - requires: assert(x >= 0 && y >= 0) - should pass
  // - ensures: assume(result.x == a && result.y == b)
  //   where __ESBMC_return_value should be replaced with 'result'
  // So: assume(result.x == 5 && result.y == 10)
  assert(result.x == 5);
  assert(result.y == 10);
  
  return 0;
}
