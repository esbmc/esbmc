#include <cassert>
#include <set>
#include <vector>

int main()
{
  int a[5] = {2, 1, 2, 3, 1};
  std::set<int> s(a, a + 5);
  assert(s.size() == 3);
  assert(s.count(1) == 1 && s.count(2) == 1 && s.count(3) == 1);
  assert(*s.begin() == 1);

  std::vector<int> v;
  v.push_back(4);
  v.push_back(4);
  v.push_back(4);
  std::set<int> t(v.begin(), v.end());
  assert(t.size() == 1);
  assert(*t.begin() == 4);
  return 0;
}
