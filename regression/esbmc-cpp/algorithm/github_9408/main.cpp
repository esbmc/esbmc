// #9408: search_n must test the last start position, last - count, and match
// at first when count == 0.
#include <algorithm>
#include <cassert>

int main()
{
  int a[3] = {1, 2, 2};
  assert(std::search_n(a, a + 3, 2, 2) == a + 1);
  assert(std::search_n(a + 1, a + 3, 2, 2) == a + 1);
  assert(std::search_n(a, a + 3, 0, 5) == a);
  assert(std::search_n(a, a + 3, 5, 2) == a + 3);
  assert(std::search_n(a, a + 3, 2, 1) == a + 3);
  return 0;
}
