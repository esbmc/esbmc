// #9367: list-initialisation selects the initializer_list constructor
// ([over.match.list]), not the bucket-count one.
#include <cassert>
#include <unordered_map>
#include <unordered_set>

int main()
{
  std::unordered_set<int> s{5};
  assert(s.size() == 1 && s.count(5) == 1);

  std::unordered_set<int> e(5);
  assert(e.empty());

  std::unordered_set<int> t{5, 6, 5};
  assert(t.size() == 2 && t.count(6) == 1);

  std::unordered_multiset<int> m{5, 5};
  assert(m.size() == 2 && m.count(5) == 2);

  std::unordered_map<int, int> u{{1, 2}, {3, 4}};
  assert(u.size() == 2 && u[3] == 4);

  std::unordered_multimap<int, int> v{{1, 2}, {1, 3}};
  assert(v.size() == 2 && v.count(1) == 2);
  return 0;
}
