// The live/raw slot boundary is crossed by insert's shift as well as by
// push_back, so the shift must construct into the one-past-the-end slot.
#include <vector>
#include <cassert>

int main()
{
  std::vector<std::vector<int>> v;
  std::vector<int> a;
  a.push_back(7);
  std::vector<int> b;
  b.push_back(9);

  v.push_back(a);
  v.insert(v.begin(), b);

  assert(v.size() == 2);
  assert(v[0][0] == 9);
  assert(v[1][0] == 7);
  return 0;
}
