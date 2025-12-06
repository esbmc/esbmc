/* Basic17_ensures_fail: Complex arithmetic expressions violation
 * Tests that violations in complex arithmetic expressions are detected
 */
#include <assert.h>

int double_sum(int x, int y)
{
  __ESBMC_ensures(__ESBMC_return_value == (x + y) * 2);
  // VIOLATION: returns wrong value
  return (x + y) * 2 - 1;  // Should be (x + y) * 2
}

int main()
{
  int r1 = double_sum(5, 10);
  return 0;
}

