#include <assert.h>

int main() {
  int x = 1; // Arbitrary value 1
  int y = 2; // Arbitrary value 2 (where x != y)

  // This should always be true: (1 * 0) == (2 * 0) -> 0 == 0 -> true
  // The expression must *not* be simplified to (x == y), which is false.
  assert((x * 0) == (y * 0));
  
  // A second case with constants on the left side
  assert((0 * x) == (0 * y));

  return 0;
}
