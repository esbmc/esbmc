/* Test: Struct return type with __ESBMC_return_value in enforce mode - PASS case
 * Expected: VERIFICATION SUCCESSFUL
 * 
 * This test verifies that __ESBMC_return_value works correctly in enforce_contracts mode
 * for struct return types. The contract should be satisfied and verification should pass.
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
  // This allows Clang to parse successfully
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
  Point result = make_point(a, b);
  
  // Contract ensures result.x == a && result.y == b
  assert(result.x == 5);
  assert(result.y == 10);
  
  return 0;
}

