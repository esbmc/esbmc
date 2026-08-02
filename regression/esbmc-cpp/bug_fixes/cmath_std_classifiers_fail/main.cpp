/* Negative variant of cmath_std_classifiers (issue #5).
 *
 * Confirms std::isinf is actually evaluated, not vacuously satisfied: a finite
 * value is not infinite, so the assertion must fail.
 *
 * Expected: VERIFICATION FAILED
 */
#include <cmath>
#include <cassert>

bool is_inf(double d)
{
  return std::isinf(d);
}

int main()
{
  double fin = 3.5;
  assert(is_inf(fin)); /* VIOLATION: 3.5 is finite */
  return 0;
}
