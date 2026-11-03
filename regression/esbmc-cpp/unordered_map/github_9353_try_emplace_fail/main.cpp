// #9353: try_emplace of a new key on a full map returned {end(), false} and
// stored nothing.
#include <cassert>
#include <unordered_map>

int main()
{
  std::unordered_map<int, int> m;
  for (int i = 0; i < 64; ++i)
    m.insert({i, i});
  assert(!m.try_emplace(64, 0).second);
  return 0;
}
