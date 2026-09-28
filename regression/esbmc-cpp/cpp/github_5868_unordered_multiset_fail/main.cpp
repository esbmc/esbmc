#include <unordered_set>
#include <cassert>

int main()
{
  std::unordered_multiset<int> m;
  m.insert(1);
  m.insert(1);
  // [unord.multiset] keeps equivalent keys, so this is 2, not the 1 a
  // uniquing container would report.
  assert(m.count(1) == 1);
  return 0;
}
