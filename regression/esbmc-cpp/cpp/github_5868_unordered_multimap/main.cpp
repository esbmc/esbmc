#include <unordered_map>
#include <cassert>

int main()
{
  std::unordered_multimap<int, int> m;
  m.insert(std::pair<int, int>(1, 10));
  m.insert(std::pair<int, int>(1, 20));
  m.insert(std::pair<int, int>(2, 30));

  // [unord.multimap]: equivalent keys are all kept.
  assert(m.size() == 3);
  assert(m.count(1) == 2);
  assert(m.count(2) == 1);
  assert(m.count(3) == 0);
  assert(!m.empty());

  int total = 0;
  for (std::unordered_multimap<int, int>::const_iterator it = m.begin();
       it != m.end(); ++it)
    total += it->second;
  assert(total == 60);

  // erase(k) removes every equivalent key.
  assert(m.erase(1) == 2);
  assert(m.size() == 1);
  assert(m.count(1) == 0);
  assert(m.find(1) == m.end());
  assert(m.find(2)->second == 30);

  m.clear();
  assert(m.empty());
  return 0;
}
