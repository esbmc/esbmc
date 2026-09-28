#include <map>
#include <cassert>

int main()
{
  std::map<int, int> m;

  auto a = m.try_emplace(1, 2);
  assert(a.second == true && a.first->second == 2 && m[1] == 2);

  // key already present: no overwrite, no insertion
  auto b = m.try_emplace(1, 99);
  assert(b.second == false && m[1] == 2);

  auto c = m.insert_or_assign(3, 4);
  assert(c.second == true && m[3] == 4);

  // key already present: assign, not insert
  auto d = m.insert_or_assign(3, 5);
  assert(d.second == false && m[3] == 5);

  return 0;
}
