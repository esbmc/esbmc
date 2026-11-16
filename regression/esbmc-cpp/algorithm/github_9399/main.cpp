// #9399: set_symmetric_difference copies the rest of range 1 once range 2 ends
#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

int main()
{
  int a[3] = {1, 2, 3};
  int b[1] = {1};
  int o[3];
  int *r = std::set_symmetric_difference(a, a + 3, b, b + 1, o);
  assert(r == o + 2 && o[0] == 2 && o[1] == 3);

  std::vector<int> va, vb, vo(3);
  va.push_back(1);
  va.push_back(2);
  va.push_back(3);
  vb.push_back(1);
  std::vector<int>::iterator vr = std::set_symmetric_difference(
    va.begin(), va.end(), vb.begin(), vb.end(), vo.begin());
  assert(vr == vo.begin() + 2 && vo[0] == 2 && vo[1] == 3);

  int c[3] = {3, 2, 1};
  int d[1] = {3};
  int *q =
    std::set_symmetric_difference(c, c + 3, d, d + 1, o, std::greater<int>());
  assert(q == o + 2 && o[0] == 2 && o[1] == 1);
  return 0;
}
