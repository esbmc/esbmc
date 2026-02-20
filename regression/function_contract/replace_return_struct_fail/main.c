/* Test: Struct return type with __ESBMC_return_value in replace mode - FAIL case
 * Expected: VERIFICATION FAILED (requires clause violation)
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
  __ESBMC_ensures(((Point*)&__ESBMC_return_value)->x == x);
  __ESBMC_ensures(((Point*)&__ESBMC_return_value)->y == y);
  
  Point p;
  p.x = x;
  p.y = y;
  return p;
}

int main()
{
  int a = -5;  // VIOLATION: violates requires clause
  int b = 10;
  Point result = make_point(a, b);
  
  return 0;
}
