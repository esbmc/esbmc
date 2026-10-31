// #9396: merge stops reading a range once it is exhausted and copies the
// rest of the other one.
#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

int main()
{
  int a[2] = {1, 5};
  int b[1] = {2};
  int o[3];
  int *r = std::merge(a, a + 2, b, b + 1, o);
  assert(r == o + 3 && o[0] == 1 && o[1] == 2 && o[2] == 5);

  int c[3] = {2, 3, 9};
  int p[4] = {0, 0, 0, 0};
  r = std::merge(a, a + 2, c, c + 1, p);
  assert(r == p + 3 && p[0] == 1 && p[1] == 2 && p[2] == 5 && p[3] == 0);

  r = std::merge(c, c + 1, a, a + 2, p);
  assert(r == p + 3 && p[0] == 1 && p[1] == 2 && p[2] == 5);

  r = std::merge(a, a, c, c + 2, p);
  assert(r == p + 2 && p[0] == 2 && p[1] == 3);

  int d[2] = {5, 1};
  int e[1] = {2};
  r = std::merge(d, d + 2, e, e + 1, o, std::greater<int>());
  assert(r == o + 3 && o[0] == 5 && o[1] == 2 && o[2] == 1);

  std::vector<int> v{1, 5};
  std::vector<int> w{2};
  std::vector<int> x(3);
  std::vector<int>::iterator q =
    std::merge(v.begin(), v.end(), w.begin(), w.end(), x.begin());
  assert(q == x.end() && x[0] == 1 && x[1] == 2 && x[2] == 5);
  return 0;
}
