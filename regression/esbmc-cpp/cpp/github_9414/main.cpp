#include <cassert>
#include <functional>
#include <set>

int main()
{
  std::set<int, std::greater<int> > s;
  s.insert(1);
  s.insert(3);
  s.insert(2);
  assert(*s.begin() == 3);
  assert(*s.rbegin() == 1);

  std::set<int, std::greater<int> > t;
  t.insert(1);
  t.insert(3);
  t.insert(5);
  assert(*t.lower_bound(4) == 3);
  assert(*t.upper_bound(3) == 1);
  assert(t.key_comp()(3, 1));

  int a[3] = {2, 7, 4};
  std::set<int, std::greater<int> > r(a, a + 3);
  assert(*r.begin() == 7);

  std::multiset<int, std::greater<int> > m;
  m.insert(1);
  m.insert(3);
  m.insert(3);
  assert(*m.begin() == 3 && m.count(3) == 2);
  return 0;
}
