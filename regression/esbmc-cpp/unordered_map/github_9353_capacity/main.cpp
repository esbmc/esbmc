// #9353: lookups of keys already present in a full map are not overflows.
#include <cassert>
#include <unordered_map>

int main()
{
  std::unordered_map<int, int> m;
  for (int i = 0; i < 64; ++i)
    m[i] = i;
  m[3] = 30;
  assert(!m.insert({4, 0}).second);
  assert(!m.try_emplace(5, 0).second);
  assert(m.size() == 64 && m[3] == 30 && m[4] == 4 && m[5] == 5);
  return 0;
}
