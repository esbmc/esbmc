// #7797: <flat_map> updates and erases keep keys and mapped values in step.
#include <flat_map>
#include <cassert>

int main()
{
  std::flat_map<int, char> m;
  m[1] = 'a';
  m[2] = 'b';
  m[3] = 'c';

  assert(!m.try_emplace(1, 'z').second);
  assert(m[1] == 'a');
  assert(m.try_emplace(4, 'd').second);
  assert(m[4] == 'd');

  assert(!m.insert_or_assign(4, 'D').second);
  assert(m[4] == 'D');

  assert(m.lower_bound(2)->first == 2);
  assert(m.upper_bound(2)->first == 3);

  assert(m.erase(2) == 1);
  assert(m.erase(2) == 0);
  assert(m.size() == 3);
  assert(m.keys()[0] == 1 && m.keys()[1] == 3 && m.keys()[2] == 4);
  assert(m.values()[1] == 'c');

  m.clear();
  assert(m.empty());
  return 0;
}
