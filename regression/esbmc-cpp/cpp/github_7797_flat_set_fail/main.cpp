// #7797: inserting a key already present must not grow the set.
#include <flat_set>
#include <cassert>

int main()
{
  std::flat_set<int> s;
  s.insert(2);
  s.insert(2);
  assert(s.size() == 2);
  return 0;
}
