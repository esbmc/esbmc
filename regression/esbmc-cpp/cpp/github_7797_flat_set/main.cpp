// #7797: <flat_set> keeps its keys sorted and unique.
#include <flat_set>
#include <cassert>
#include <utility>

int main()
{
  std::flat_set<int> s;
  assert(s.empty());

  assert(s.insert(5).second);
  assert(s.insert(1).second);
  assert(s.insert(3).second);
  // A duplicate is rejected and the size is unchanged.
  assert(!s.insert(3).second);
  assert(s.size() == 3);

  assert(s.contains(1) && s.contains(3) && s.contains(5));
  assert(!s.contains(2));
  assert(s.count(3) == 1 && s.count(4) == 0);

  // Sorted, whatever the insertion order was.
  std::flat_set<int>::iterator it = s.begin();
  assert(*it == 1);
  ++it;
  assert(*it == 3);
  ++it;
  assert(*it == 5);
  ++it;
  assert(it == s.end());

  assert(*s.find(3) == 3);
  assert(s.find(4) == s.end());
  assert(*s.lower_bound(2) == 3);
  assert(*s.upper_bound(3) == 5);

  assert(s.erase(3) == 1);
  assert(s.erase(3) == 0);
  assert(s.size() == 2);
  assert(!s.contains(3));
  assert(*s.begin() == 1);

  std::vector<int> out = std::move(s).extract();
  assert(out.size() == 2 && out[0] == 1 && out[1] == 5);
  assert(s.empty());

  s.replace(std::move(out));
  assert(s.size() == 2 && s.contains(5));

  s.clear();
  assert(s.empty());
  return 0;
}
