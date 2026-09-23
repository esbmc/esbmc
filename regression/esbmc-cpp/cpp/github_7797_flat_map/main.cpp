// #7797: <flat_map> keeps keys and mapped values sorted in step.
#include <flat_map>
#include <cassert>
#include <utility>

int main()
{
  std::flat_map<int, char> m;
  assert(m.empty());

  m[3] = 'c';
  m[1] = 'a';
  assert(m.insert(std::pair<int, char>(2, 'b')).second);
  assert(!m.insert(std::pair<int, char>(2, 'x')).second);
  assert(m.size() == 3);

  assert(m[1] == 'a' && m[2] == 'b' && m[3] == 'c');
  assert(m.at(2) == 'b');
  assert(m.contains(3) && !m.contains(4));
  assert(m.count(1) == 1 && m.count(9) == 0);

  // The keys are sorted, and the values moved with them.
  assert(m.keys()[0] == 1 && m.keys()[1] == 2 && m.keys()[2] == 3);
  assert(m.values()[0] == 'a' && m.values()[1] == 'b' && m.values()[2] == 'c');

  std::flat_map<int, char>::iterator it = m.find(2);
  assert(it != m.end());
  assert(it->first == 2 && it->second == 'b');
  ++it;
  assert(it->first == 3);

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
