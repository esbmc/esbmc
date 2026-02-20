/* Test: Struct return type with __ESBMC_return_value in enforce mode - FAIL case
 * Expected: VERIFICATION FAILED
 * 
 * This test verifies that __ESBMC_return_value correctly detects contract violations
 * in enforce_contracts mode for struct return types. The contract should be violated
 * and verification should fail.
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
  __ESBMC_ensures(((Point*)&__ESBMC_return_value)->y == y);  // This should detect violation!
  
  Point p;
  p.x = x;
  p.y = y + 1;  // VIOLATION: wrong y value (should be y, not y+1)
  return p;
}

int main()
{
  int a = 5;
  int b = 10;
  Point result = make_point(a, b);
  
  // Contract ensures result.x == a && result.y == b
  // But result.y is actually 11, so verification should FAIL
  assert(result.x == 5);
  assert(result.y == 10);  // This will fail because result.y == 11
  
  return 0;
}

