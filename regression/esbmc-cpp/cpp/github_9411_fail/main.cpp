#include <algorithm>
#include <cassert>

bool by_tens(int x, int y)
{
  return x / 10 < y / 10;
}

int main()
{
  int a[3] = {20, 21, 10};
  std::inplace_merge(a, a + 2, a + 3, by_tens);
  assert(!(a[0] == 10 && a[1] == 20 && a[2] == 21));
  return 0;
}
