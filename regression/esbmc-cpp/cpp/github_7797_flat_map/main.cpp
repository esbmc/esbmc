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
  return 0;
}
