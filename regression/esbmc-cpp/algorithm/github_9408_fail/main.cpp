// #9408
#include <algorithm>
#include <cassert>

int main()
{
  int a[3] = {1, 2, 2};
  int *r = std::search_n(a, a + 3, 2, 2);
  assert(!(r == a + 1));
  return 0;
}
