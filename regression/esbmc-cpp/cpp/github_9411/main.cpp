#include <algorithm>
#include <cassert>
#include <list>

bool by_tens(int x, int y)
{
  return x / 10 < y / 10;
}

int main()
{
  int a[3] = {20, 21, 10};
  std::inplace_merge(a, a + 2, a + 3, by_tens);
  assert(a[0] == 10 && a[1] == 20 && a[2] == 21);

  int b[5] = {11, 30, 10, 12, 31};
  std::inplace_merge(b, b + 2, b + 5, by_tens);
  assert(b[0] == 11 && b[1] == 10 && b[2] == 12);
  assert(b[3] == 30 && b[4] == 31);

  int c[4] = {1, 4, 2, 3};
  std::inplace_merge(c, c + 2, c + 4);
  assert(c[0] == 1 && c[1] == 2 && c[2] == 3 && c[3] == 4);

  int d[2] = {2, 1};
  std::inplace_merge(d, d + 2, d + 2);
  assert(d[0] == 2 && d[1] == 1);

  std::list<int> l;
  l.push_back(3);
  l.push_back(1);
  std::list<int>::iterator mid = l.begin();
  ++mid;
  std::inplace_merge(l.begin(), mid, l.end());
  assert(l.front() == 1 && l.back() == 3);
  return 0;
}
