// #9408
#include <algorithm>
#include <cassert>

bool less_than(int x, int y)
{
  return x < y;
}

int main()
{
  int a[3] = {5, 1, 0};
  int *r = std::search_n(a, a + 3, 2, 2, less_than);
  assert(!(r == a + 1));
  return 0;
}
