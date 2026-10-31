// #9408: predicate overload.
#include <algorithm>
#include <cassert>

bool less_than(int x, int y)
{
  return x < y;
}

int main()
{
  int a[3] = {5, 1, 0};
  assert(std::search_n(a, a + 3, 2, 2, less_than) == a + 1);
  assert(std::search_n(a, a + 3, 0, 2, less_than) == a);
  assert(std::search_n(a, a + 3, 5, 9, less_than) == a + 3);
  return 0;
}
