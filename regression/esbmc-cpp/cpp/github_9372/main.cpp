// github #9372: map::insert returns an iterator to the inserted element and
// map::erase(key) returns the number of elements erased.
#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  std::pair<std::map<int, int>::iterator, bool> r = m.insert({5, 50});
  assert(r.second && r.first != m.end() && r.first->first == 5);
  r = m.insert({3, 30});
  assert(r.second && r.first->first == 3 && r.first->second == 30);
  assert(m.erase(3) == 1);
  assert(m.erase(4) == 0);

  std::map<int, int> n;
  n[1] = 10;
  std::map<int, int>::iterator h = n.insert(n.end(), {5, 50});
  assert(h != n.end() && h->first == 5 && h->second == 50);
  h = n.insert(n.begin(), {3, 30});
  assert(h->first == 3 && h->second == 30);
  return 0;
}
