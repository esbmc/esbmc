#include <unordered_set>
#include <cassert>

int main()
{
  std::unordered_multiset<int> m;
  m.insert(1);
  m.insert(1);
  m.insert(2);

  // [unord.multiset]: equivalent keys are all kept.
  assert(m.size() == 3);
  assert(m.count(1) == 2);
  assert(m.count(2) == 1);
  assert(m.count(3) == 0);
  assert(!m.empty());

  // erase(k) removes every equivalent key.
  assert(m.erase(1) == 2);
  assert(m.size() == 1);
  assert(m.count(1) == 0);
  assert(m.count(2) == 1);

  assert(m.find(2) != m.end());
  assert(m.find(9) == m.end());

  int total = 0;
  for (std::unordered_multiset<int>::iterator it = m.begin(); it != m.end(); ++it)
    total += *it;
  assert(total == 2);

  m.clear();
  assert(m.empty());
  return 0;
}
