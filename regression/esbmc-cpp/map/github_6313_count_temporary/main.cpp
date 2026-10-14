// github #6313: std::map::count must accept a temporary key and work on a const
// map. The model declared count(key_type&) (non-const ref), so count(1) failed
// to compile. It now matches the standard: count(const key_type&) const.
#include <cassert>
#include <map>

int query(const std::map<int, int> &m, int k)
{
  return m.count(k); // count on a const map
}

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  m[2] = 20;

  assert(m.count(1) == 1);  // temporary key
  assert(m.count(3) == 0);
  assert(query(m, 2) == 1); // const map
  m.erase(1);
  assert(m.count(1) == 0);
  return 0;
}
