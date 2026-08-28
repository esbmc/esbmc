#include <unordered_map>
#include <cassert>

int main()
{
  std::unordered_multimap<int, int> m;
  m.insert(std::pair<int, int>(1, 10));
  m.insert(std::pair<int, int>(1, 20));

  // unordered_map would collapse these to one; a multimap must not.
  assert(m.size() == 1);
  return 0;
}
