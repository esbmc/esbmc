// Negative variant: after make_heap the root holds the maximum (5), so
// asserting it equals a non-maximum must fail.
#include <algorithm>
#include <cassert>

int main()
{
  int a[5] = {3, 1, 4, 1, 5};
  std::make_heap(a, a + 5);
  assert(a[0] == 3);
  return 0;
}
